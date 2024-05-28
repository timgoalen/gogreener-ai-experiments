export const testData: string = `
// greener/actor.go

package greener

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// trimBearer removes the "Bearer " prefix from an authorization header, case-insensitively, and trims any surrounding whitespace.
func trimBearer(authHeader string) string {
	// Define the prefix in a standard case.
	prefix := "bearer"
	// Convert the header to lowercase to ensure case-insensitive comparison.
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(authHeader)), prefix) {
		// Trim the prefix and any surrounding whitespace
		return strings.TrimSpace(strings.TrimSpace(authHeader)[len(prefix):])
	}
	return strings.TrimSpace(authHeader)
}

func PollForHealth(url string, timeout time.Duration, retryTimeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil && resp.StatusCode == http.StatusOK {
			// fmt.Println("Server is healthy and ready.")
			return nil
		}
		// if err != nil {
		// 	fmt.Printf("Failed to reach server: %v\n", err)
		// } else {
		// 	fmt.Printf("Server not ready, status code: %d\n", resp.StatusCode)
		// }
		time.Sleep(retryTimeout) // sleep before retrying
	}
	return fmt.Errorf("server at %s not ready within %v", url, timeout)
}

// handle is a generic function that abstracts the common pattern of decoding an HTTP request,
// processing it using a provided function that might return an error, and encoding the result back to HTTP response.
func ActorHandleCall[M any, T any](w http.ResponseWriter, r *http.Request, process func(context.Context, string, M) (T, error)) {
	var message M
	credentials := trimBearer(r.Header.Get("Authorization"))
	ctx := r.Context()
	timeoutHeader := r.Header.Get("X-Timeout-Ms")
	if timeoutHeader != "" {
		// Parse timeout from header
		timeoutMs, err := strconv.Atoi(timeoutHeader)
		if err != nil {
			http.Error(w, "Invalid timeout value", http.StatusBadRequest)
			return
		}
		c, cancel := context.WithTimeout(ctx, time.Duration(timeoutMs)*time.Millisecond)
		defer cancel()
		ctx = c
	}
	if err := json.NewDecoder(r.Body).Decode(&message); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	response, err := process(ctx, credentials, message)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error processing request: %v", err), http.StatusInternalServerError)
		return
	}
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

// RemoteCall is a generic function that sends a request and expects a response, handling errors and HTTP communication.
func ActorRemoteCall[R any, T any](client *http.Client, serverURL, endpoint string, ctx context.Context, credentials string, requestData R) (T, error) {
	var responseData T
	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return responseData, fmt.Errorf("error marshalling request data: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", serverURL+"/"+endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return responseData, fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+credentials)
	deadline, ok := ctx.Deadline()
	if ok {
		timeout := time.Until(deadline)
		req.Header.Set("X-Timeout-Ms", fmt.Sprintf("%d", timeout.Milliseconds()))
	}
	response, err := client.Do(req)
	if err != nil {
		return responseData, fmt.Errorf("error sending request to '%s': %w", endpoint, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return responseData, fmt.Errorf("received non-OK HTTP status from '%s': %s", endpoint, response.Status)
	}
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return responseData, fmt.Errorf("error reading response body from '%s': %w", endpoint, err)
	}
	if err := json.Unmarshal(body, &responseData); err != nil {
		return responseData, fmt.Errorf("error unmarshalling response from '%s': %w", endpoint, err)
	}
	return responseData, nil
}



// greener/content.go

package greener

import (
	"bytes"
	"compress/gzip"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"github.com/andybalholm/brotli"
	"net/http"
	"strconv"
	"strings"
	"sync"
)

// ContentHandler handles brotli and gzip compression of content as well as generating a hash so that smallet content possible can be served to the client based on the request Content-Encoding. The response is set with 1 year cache max age with the intention that the URL the handler is registered with includes the content hash so that if the content changes, so would the URL.
type ContentHandler interface {
	Hash() string
	ServeHTTP(http.ResponseWriter, *http.Request)
}

type contentHandler struct {
	hash          string
	contentType   string
	content       []byte
	gzipContent   []byte
	brotliContent []byte
	logger        Logger
	cacheSeconds  int
}

// NewContentHandler returns a struct containing a hash of the content as well as gzip and brotli compressed content encodings. It implements http.Handler for serving the most appropriate content encoding based on the request.
func NewContentHandler(logger Logger, content []byte, contentType, salt string, cacheSeconds int) ContentHandler {
	// Hash the content with salt
	hash := hashContentWithSalt(content, salt) // New hashing function

	// Compress the content
	originalBytes := []byte(content)
	gzipContent, brotliContent, err := compressContent(originalBytes)
	if err != nil {
		logger.Logf("Failed to compress the content: %v", err)
	}

	// Check if compressed versions are actually shorter
	if gzipContent == nil || len(gzipContent) >= len(originalBytes) {
		gzipContent = nil
	}
	if brotliContent == nil || len(brotliContent) >= len(originalBytes) {
		brotliContent = nil
	}

	return &contentHandler{
		hash:          hash,
		contentType:   contentType,
		content:       originalBytes,
		gzipContent:   gzipContent,
		brotliContent: brotliContent,
		cacheSeconds:  cacheSeconds,
	}
}

func (c *contentHandler) Hash() string {
	return c.hash
}

func hashContentWithSalt(content []byte, salt string) string {
	hmac := hmac.New(sha256.New, []byte(salt))
	hmac.Write(content)
	sum := hmac.Sum(nil)
	return base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString(sum)
}

func compressContent(content []byte) ([]byte, []byte, error) {
	var gzipBuffer, brotliBuffer bytes.Buffer
	var wg sync.WaitGroup
	var errGzip, errBrotli error

	wg.Add(1)
	go func() {
		defer wg.Done()
		gzipWriter := gzip.NewWriter(&gzipBuffer)
		if _, err := gzipWriter.Write(content); err != nil {
			errGzip = err
			return
		}
		errGzip = gzipWriter.Close()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		brotliWriter := brotli.NewWriterLevel(&brotliBuffer, brotli.BestCompression)
		if _, err := brotliWriter.Write(content); err != nil {
			errBrotli = err
			return
		}
		errBrotli = brotliWriter.Close()
	}()

	wg.Wait()

	if errGzip != nil && errBrotli != nil {
		return nil, nil, fmt.Errorf("both Gzip and Brotli compression failed: %v\n%v", errGzip, errBrotli)
	} else if errGzip == nil && errBrotli == nil {
		return gzipBuffer.Bytes(), brotliBuffer.Bytes(), nil
	} else if errBrotli != nil {
		return gzipBuffer.Bytes(), nil, errBrotli
	} else {
		return nil, brotliBuffer.Bytes(), errGzip
	}
}

func (c *contentHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Set ETag header
	etag := "\"" + c.Hash() + "\"" // Adding quotes around the ETag as per the HTTP ETag format
	w.Header().Set("ETag", etag)
	// Check if the client has sent the If-None-Match header and compare the ETag
	if match := r.Header.Get("If-None-Match"); match != "" {
		if match = strings.TrimSpace(match); EtagMatch(match, etag) {
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}

	// Otherwise choose based partly on Accept-Encoding
	acceptEncoding := r.Header.Get("Accept-Encoding")

	// Default to sending original content unless a smaller, acceptable version is available
	bestEncoding := "identity"
	bestContent := c.content
	var bestContentSize int = len(c.content)

	encodings := ParseEncodings(acceptEncoding)

	// Determine if an encoding is acceptable based on q-values
	isAcceptable := func(encoding string) bool {
		q, exists := encodings[encoding]
		return exists && q > 0
	}

	// Build a list of acceptable encodings with their corresponding content sizes
	acceptableEncodings := []struct {
		encodingName string
		content      []byte
	}{
		{"identity", c.content},
	}

	if isAcceptable("gzip") && c.gzipContent != nil {
		acceptableEncodings = append(acceptableEncodings, struct {
			encodingName string
			content      []byte
		}{"gzip", c.gzipContent})
	}

	if isAcceptable("br") && c.brotliContent != nil {
		acceptableEncodings = append(acceptableEncodings, struct {
			encodingName string
			content      []byte
		}{"br", c.brotliContent})
	}

	// Select the smallest acceptable encoding
	for _, encoding := range acceptableEncodings {
		if len(encoding.content) < bestContentSize {
			bestEncoding = encoding.encodingName
			bestContent = encoding.content
			bestContentSize = len(encoding.content)
		}
	}

	// Apply wildcard logic if no other encoding than identity is chosen and a wildcard is present
	wildcardQ, wildcardPresent := encodings["*"]
	if bestEncoding == "identity" && wildcardPresent && wildcardQ > 0 && c.gzipContent != nil {
		bestEncoding = "gzip"
		bestContent = c.gzipContent
	}

	// Set Content-Encoding header if not using the identity encoding
	if bestEncoding != "identity" {
		w.Header().Set("Content-Encoding", bestEncoding)
	}
	w.Header().Set("Content-Type", c.contentType)
	if c.cacheSeconds > 0 {
		w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=%d", c.cacheSeconds))
	}

	// Write the content
	contentLength := strconv.Itoa(len(bestContent))
	w.Header().Set("Content-Length", contentLength)
	w.Write(bestContent)
}

func EtagMatch(header, etag string) bool {
	etags := strings.Split(header, ",")
	for _, e := range etags {
		trimmedEtag := strings.TrimSpace(e)
		// Remove surrounding quotes for a standardized comparison
		if len(trimmedEtag) >= 2 && trimmedEtag[0] == '"' && trimmedEtag[len(trimmedEtag)-1] == '"' {
			trimmedEtag = trimmedEtag[1 : len(trimmedEtag)-1]
		}
		// Compare without quotes
		if trimmedEtag == strings.Trim(etag, "\"") {
			return true
		}
	}
	return false
}

// Helper function to parse the Accept-Encoding header
func ParseEncodings(header string) map[string]float64 {
	encodings := make(map[string]float64)
	for _, part := range strings.Split(header, ",") {
		pieces := strings.Split(strings.TrimSpace(part), ";")
		encoding := strings.TrimSpace(pieces[0])
		qValue := 1.0 // Default qValue for encodings not specifying q-value
		// This doesn't track the highest q value, it just tracks the last one that isn't 0. In HTTP headers I understand the last valid occurrence of the item takes precedence.
		if len(pieces) > 1 {
			qPart := strings.TrimSpace(pieces[1])
			if strings.HasPrefix(qPart, "q=") {
				var parsedQ float64
				if _, err := fmt.Sscanf(qPart[2:], "%f", &parsedQ); err == nil && parsedQ >= 0 && parsedQ <= 1 {
					encodings[encoding] = parsedQ
				}
			}
		} else {
			encodings[encoding] = qValue
		}
	}
	return encodings
}



// greener/db.go

package greener

import (
	"context"
	"database/sql"
	"fmt"
	"net/url"
	"runtime"
	"strings"
	"sync"
	"time"
)

func setupSqlitePragmas(db *sql.DB) error {
	for key, value := range sqlitePragmas {
		if strings.HasPrefix(key, "_") {
			// Strip the leading "_" and set the pragma
			pragma := strings.TrimPrefix(key, "_") + " = " + value
			if _, err := db.Exec("PRAGMA " + pragma); err != nil {
				return err
			}
		}
	}

	return nil
}

func newSQLiteConnectionURL(path string) string {
	connectionUrlParams := make(url.Values)
	for key, value := range sqlitePragmas {
		// For connection string, include keys as they are
		connectionUrlParams.Add(key, value)
	}
	return "file:" + path + "?" + connectionUrlParams.Encode()
}

type writeRequest struct {
	resp chan error
	fn   func(WriteDBHandler) error
}

type ReadDBHandler interface {
	ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row
}

type WriteDBHandler interface {
	ExecContext(ctx context.Context, query string, args ...interface{}) (*resultWrapper, error)
	QueryContext(ctx context.Context, query string, args ...interface{}) (*rowsWrapper, error)
	QueryRowContext(ctx context.Context, query string, args ...interface{}) *rowWrapper
}

type DBModifier interface {
	Write(func(WriteDBHandler) error) error
}

type DB interface {
	ReadDBHandler
	DBModifier
}

type BatchDB struct {
	ReadDBHandler
	readDB        *sql.DB
	writeDB       *sql.DB
	writeDBLock   sync.Mutex
	writeRequests chan writeRequest
	flushTimeout  time.Duration
}

func NewBatchDB(path string, flushTimeout time.Duration) (*BatchDB, error) {

	connectionURL := newSQLiteConnectionURL(path)
	// fmt.Println(connectionURL)

	writeDB, err := sql.Open(SqlDriver, connectionURL+"&mode=rwc")
	if err != nil {
		return nil, err
	}
	writeDB.SetMaxOpenConns(1)
	if err = setupSqlitePragmas(writeDB); err != nil {
		return nil, err
	}

	// Put the read connection into literally read only mode.
	ReadDB, err := sql.Open(SqlDriver, connectionURL+"&mode=ro")
	if err != nil {
		return nil, err
	}
	maxConns := 4
	if n := runtime.NumCPU(); n > maxConns {
		maxConns = n
	}
	ReadDB.SetMaxOpenConns(maxConns)
	if err = setupSqlitePragmas(ReadDB); err != nil {
		return nil, err
	}

	db := &BatchDB{
		ReadDBHandler: ReadDB,
		readDB:        ReadDB,
		writeDB:       writeDB,
		writeRequests: make(chan writeRequest),
		flushTimeout:  flushTimeout,
	}

	go db.batchProcessor()
	return db, nil
}

func (db *BatchDB) batchProcessor() {
	var requests []writeRequest
	var currentTx *sql.Tx
	timer := time.NewTicker(db.flushTimeout * time.Millisecond)

	for {
		select {
		case req := <-db.writeRequests:
			if len(requests) == 0 {
				var err error
				currentTx, err = db.writeDB.Begin()
				if err != nil {
					req.resp <- err
					continue
				}
			}
			txWrapper := &txWrapper{tx: currentTx}
			err := req.fn(txWrapper)
			if err != nil {
				// fmt.Printf("Rolling back: %v\n", err)
				if txWrapper.err == nil {
					txWrapper.Abort(err)
				}
				// The original error is returned to the caller
				req.resp <- err
			}
			if txWrapper.err != nil {
				for _, r := range requests {
					// All the earlier goroutines get a standard message
					r.resp <- fmt.Errorf("transaction aborted")
				}
				requests = requests[:0]
				continue
			}
			requests = append(requests, req)
		case <-timer.C:
			if len(requests) > 0 {
				// fmt.Printf("Committing\n")
				commitErr := currentTx.Commit()
				currentTx = nil
				for _, req := range requests {
					req.resp <- commitErr
				}
				requests = requests[:0]
			}
		}
	}
}

func (db *BatchDB) Write(fn func(WriteDBHandler) error) error {
	respChan := make(chan error)
	req := writeRequest{
		fn:   fn,
		resp: respChan,
	}
	db.writeRequests <- req
	err := <-respChan
	return err
}

func (db *BatchDB) Close() error {
	rerr := db.readDB.Close()
	db.writeDBLock.Lock()
	defer db.writeDBLock.Unlock()
	werr := db.writeDB.Close()
	if rerr != nil || werr != nil {
		return fmt.Errorf("error closing connections. Write DB Err: %v. Read DB err: %v.\n", werr, rerr)
	}
	return nil
}



// greener/fts.go

// rm search_engine.db ; go run -tags "sqlite_fts5" main.go

package greener

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"sort"
	"strings"
)

type FTS struct {
	db DB
}

type Facet struct {
	Name  string
	Value string
}

type FacetValueCount struct {
	Value string
	Count int
}

type FacetCount struct {
	Name   string
	Values []FacetValueCount
}

func NewFTS(ctx context.Context, db DB) (*FTS, error) {
	// Ensure the FTS table and facet tables exist
	// _, err = d.ExecContext(ctx, "INSERT INTO document_facets (document_id, facet_id) VALUES (?, ?)", docid, facetID)
	// search_test.go:64: Error adding facets: Could not insert document_facet: SQL logic error: foreign key mismatch - "document_facets" referencing "documents" (1)

	queries := []string{
		\`CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(content, docid UNINDEXED);\`,
		\`CREATE TABLE IF NOT EXISTS facets (id INTEGER PRIMARY KEY, name TEXT, value TEXT, UNIQUE(name, value));\`,
		// \`CREATE TABLE IF NOT EXISTS document_facets (document_id TEXT, facet_id INTEGER, FOREIGN KEY(document_id) REFERENCES documents(docid), FOREIGN KEY(facet_id) REFERENCES facets(id));\`,
		\`CREATE TABLE IF NOT EXISTS document_facets (document_id TEXT, facet_id INTEGER, FOREIGN KEY(facet_id) REFERENCES facets(id));\`,
	}
	for _, query := range queries {
		err := db.Write(func(d WriteDBHandler) error {
			if _, err := d.ExecContext(ctx, query); err != nil {
				return err
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return &FTS{db: db}, nil
}

func (se *FTS) Put(ctx context.Context, docid string, reader io.Reader) error {
	content, err := ioutil.ReadAll(reader)
	if err != nil {
		return err
	}

	err = se.db.Write(func(d WriteDBHandler) error {
		// I think we need to do the two operations separately because of a limitation in FT5 virtual tables, but should check this again.

		// Attempt to update the document first.
		result, err := d.ExecContext(ctx, "UPDATE documents SET content = ? WHERE docid = ?", string(content), docid)
		if err != nil {
			return err
		}
		// Check if the update operation affected any rows.
		rowsAffected, err := result.RowsAffected()
		if err != nil {
			return err
		}

		// If no rows were affected by the update, the document does not exist and needs to be inserted.
		if rowsAffected == 0 {
			_, err = d.ExecContext(ctx, "INSERT INTO documents(docid, content) VALUES(?, ?)", docid, string(content))
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	return nil
}

func (se *FTS) Delete(ctx context.Context, docid string) error {
	err := se.db.Write(func(d WriteDBHandler) error {
		_, err := d.ExecContext(ctx, "DELETE FROM documents WHERE docid = ?", docid)
		return err
	})
	if err != nil {
		return err
	}
	return nil
}

func (se *FTS) Get(ctx context.Context, docid string) (io.Reader, error) {
	var content string
	row := se.db.QueryRowContext(ctx, "SELECT content FROM documents WHERE docid = ?", docid)
	if err := row.Scan(&content); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, errors.New("document not found")
		}
		return nil, err
	}
	return strings.NewReader(content), nil
}

func (se *FTS) Search(ctx context.Context, query string) ([]map[string]string, error) {
	rows, err := se.db.QueryContext(ctx, "SELECT docid, snippet(documents, 0, '<b>', '</b>', '...', 64) FROM documents WHERE documents MATCH ? ORDER BY rank", query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []map[string]string
	for rows.Next() {
		var docid, snippet string
		if err := rows.Scan(&docid, &snippet); err != nil {
			return nil, err
		}
		results = append(results, map[string]string{"docid": docid, "content": snippet})
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}

func (se *FTS) AddFacets(ctx context.Context, docid string, facets []Facet) error {
	for _, facet := range facets {
		err := se.db.Write(func(d WriteDBHandler) error {
			result, err := d.ExecContext(ctx, "INSERT INTO facets (name, value) VALUES (?, ?) ON CONFLICT(name, value) DO NOTHING", facet.Name, facet.Value)
			if err != nil {
				return fmt.Errorf("could not insert facet: %v\n", err)
			}

			facetID, err := result.LastInsertId()
			if err != nil || facetID == 0 {
				err = d.QueryRowContext(ctx, "SELECT id FROM facets WHERE name = ? AND value = ?", facet.Name, facet.Value).Scan(&facetID)
				if err != nil {
					return fmt.Errorf("could not get facet ID: %v\n", err)
				}
			}

			_, err = d.ExecContext(ctx, "INSERT INTO document_facets (document_id, facet_id) VALUES (?, ?)", docid, facetID)
			if err != nil {
				return fmt.Errorf("could not insert document_facet: %v\n", err)
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (se *FTS) GetFacetCounts(ctx context.Context, docIDs []string) ([]FacetCount, error) {
	if len(docIDs) == 0 {
		return []FacetCount{}, nil
	}

	inParams := strings.Repeat("?,", len(docIDs)-1) + "?"
	query := fmt.Sprintf("SELECT f.name, f.value, COUNT(*) as count FROM document_facets df JOIN facets f ON df.facet_id = f.id WHERE df.document_id IN (%s) GROUP BY f.name, f.value ORDER BY f.name, count DESC", inParams)

	rows, err := se.db.QueryContext(ctx, query, stringsToInterfaces(docIDs)...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	facetCountsMap := make(map[string][]FacetValueCount)
	for rows.Next() {
		var name, value string
		var count int
		if err := rows.Scan(&name, &value, &count); err != nil {
			return nil, err
		}

		facetCountsMap[name] = append(facetCountsMap[name], FacetValueCount{Value: value, Count: count})
	}

	var facetCounts []FacetCount
	for name, values := range facetCountsMap {
		facetCounts = append(facetCounts, FacetCount{Name: name, Values: values})
	}

	return facetCounts, nil
}

func SortFacetsByTotalDocCount(facets []FacetCount) {
	sort.Slice(facets, func(i, j int) bool {
		iTotal, jTotal := 0, 0
		for _, v := range facets[i].Values {
			iTotal += v.Count
		}
		for _, v := range facets[j].Values {
			jTotal += v.Count
		}
		return iTotal > jTotal
	})
}

func OrderFacetsByNames(facets []FacetCount, order []string) []FacetCount {
	orderMap := make(map[string]int)
	for i, name := range order {
		orderMap[name] = i
	}

	sortedFacets := make([]FacetCount, len(facets))
	copy(sortedFacets, facets)

	sort.SliceStable(sortedFacets, func(i, j int) bool {
		iOrder, iFound := orderMap[sortedFacets[i].Name]
		jOrder, jFound := orderMap[sortedFacets[j].Name]

		if iFound && jFound {
			return iOrder < jOrder
		}
		if iFound {
			return true
		}
		if jFound {
			return false
		}
		return sortedFacets[i].Name < sortedFacets[j].Name
	})

	return sortedFacets
}

func stringsToInterfaces(strings []string) []interface{} {
	interfaces := make([]interface{}, len(strings))
	for i, s := range strings {
		interfaces[i] = s
	}
	return interfaces
}

// Extracts docIDs from search results
func GetDocIDsFromSearchResults(results []map[string]string) []string {
	var docIDs []string
	for _, result := range results {
		docIDs = append(docIDs, result["docid"])
	}
	return docIDs
}



// greener/go.mod

module github.com/thejimmyg/greener

go 1.19

require (
	github.com/Kodeworks/golang-image-ico v0.0.0-20141118225523-73f0f4cfade9
	github.com/andybalholm/brotli v1.1.0
	github.com/mattn/go-sqlite3 v1.14.22
	github.com/yuin/goldmark v1.7.1
	golang.org/x/image v0.15.0
	modernc.org/sqlite v1.29.5
)

require (
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/hashicorp/golang-lru/v2 v2.0.7 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/ncruces/go-strftime v0.1.9 // indirect
	github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec // indirect
	golang.org/x/sys v0.19.0 // indirect
	modernc.org/gc/v3 v3.0.0-20240107210532-573471604cb6 // indirect
	modernc.org/libc v1.41.0 // indirect
	modernc.org/mathutil v1.6.0 // indirect
	modernc.org/memory v1.8.0 // indirect
	modernc.org/strutil v1.2.0 // indirect
	modernc.org/token v1.1.0 // indirect
)



// greener/go.sum

github.com/Kodeworks/golang-image-ico v0.0.0-20141118225523-73f0f4cfade9 h1:1ltqoej5GtaWF8jaiA49HwsZD459jqm9YFz9ZtMFpQA=
github.com/Kodeworks/golang-image-ico v0.0.0-20141118225523-73f0f4cfade9/go.mod h1:7uhhqiBaR4CpN0k9rMjOtjpcfGd6DG2m04zQxKnWQ0I=
github.com/andybalholm/brotli v1.1.0 h1:eLKJA0d02Lf0mVpIDgYnqXcUn0GqVmEFny3VuID1U3M=
github.com/andybalholm/brotli v1.1.0/go.mod h1:sms7XGricyQI9K10gOSf56VKKWS4oLer58Q+mhRPtnY=
github.com/dustin/go-humanize v1.0.1 h1:GzkhY7T5VNhEkwH0PVJgjz+fX1rhBrR7pRT3mDkpeCY=
github.com/dustin/go-humanize v1.0.1/go.mod h1:Mu1zIs6XwVuF/gI1OepvI0qD18qycQx+mFykh5fBlto=
github.com/google/pprof v0.0.0-20221118152302-e6195bd50e26 h1:Xim43kblpZXfIBQsbuBVKCudVG457BR2GZFIz3uw3hQ=
github.com/google/uuid v1.6.0 h1:NIvaJDMOsjHA8n1jAhLSgzrAzy1Hgr+hNrb57e+94F0=
github.com/google/uuid v1.6.0/go.mod h1:TIyPZe4MgqvfeYDBFedMoGGpEw/LqOeaOT+nhxU+yHo=
github.com/hashicorp/golang-lru/v2 v2.0.7 h1:a+bsQ5rvGLjzHuww6tVxozPZFVghXaHOwFs4luLUK2k=
github.com/hashicorp/golang-lru/v2 v2.0.7/go.mod h1:QeFd9opnmA6QUJc5vARoKUSoFhyfM2/ZepoAG6RGpeM=
github.com/mattn/go-isatty v0.0.20 h1:xfD0iDuEKnDkl03q4limB+vH+GxLEtL/jb4xVJSWWEY=
github.com/mattn/go-isatty v0.0.20/go.mod h1:W+V8PltTTMOvKvAeJH7IuucS94S2C6jfK/D7dTCTo3Y=
github.com/mattn/go-sqlite3 v1.14.22 h1:2gZY6PC6kBnID23Tichd1K+Z0oS6nE/XwU+Vz/5o4kU=
github.com/mattn/go-sqlite3 v1.14.22/go.mod h1:Uh1q+B4BYcTPb+yiD3kU8Ct7aC0hY9fxUwlHK0RXw+Y=
github.com/ncruces/go-strftime v0.1.9 h1:bY0MQC28UADQmHmaF5dgpLmImcShSi2kHU9XLdhx/f4=
github.com/ncruces/go-strftime v0.1.9/go.mod h1:Fwc5htZGVVkseilnfgOVb9mKy6w1naJmn9CehxcKcls=
github.com/pmezard/go-difflib v1.0.0 h1:4DBwDE0NGyQoBHbLQYPwSUPoCMWR5BEzIk/f1lZbAQM=
github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec h1:W09IVJc94icq4NjY3clb7Lk8O1qJ8BdBEF8z0ibU0rE=
github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec/go.mod h1:qqbHyh8v60DhA7CoWK5oRCqLrMHRGoxYCSS9EjAz6Eo=
github.com/yuin/goldmark v1.7.1 h1:3bajkSilaCbjdKVsKdZjZCLBNPL9pYzrCakKaf4U49U=
github.com/yuin/goldmark v1.7.1/go.mod h1:uzxRWxtg69N339t3louHJ7+O03ezfj6PlliRlaOzY1E=
golang.org/x/image v0.15.0 h1:kOELfmgrmJlw4Cdb7g/QGuB3CvDrXbqEIww/pNtNBm8=
golang.org/x/image v0.15.0/go.mod h1:HUYqC05R2ZcZ3ejNQsIHQDQiwWM4JBqmm6MKANTp4LE=
golang.org/x/mod v0.14.0 h1:dGoOF9QVLYng8IHTm7BAyWqCqSheQ5pYWGhzW00YJr0=
golang.org/x/sys v0.6.0/go.mod h1:oPkhp1MJrh7nUepCBck5+mAzfO9JrbApNNgaTdGDITg=
golang.org/x/sys v0.19.0 h1:q5f1RH2jigJ1MoAWp2KTp3gm5zAGFUTarQZ5U386+4o=
golang.org/x/sys v0.19.0/go.mod h1:/VUhepiaJMQUp4+oa/7Zr1D23ma6VTLIYjOOTFZPUcA=
golang.org/x/tools v0.17.0 h1:FvmRgNOcs3kOa+T20R1uhfP9F6HgG2mfxDv1vrx1Htc=
modernc.org/fileutil v1.3.0 h1:gQ5SIzK3H9kdfai/5x41oQiKValumqNTDXMvKo62HvE=
modernc.org/gc/v3 v3.0.0-20240107210532-573471604cb6 h1:5D53IMaUuA5InSeMu9eJtlQXS2NxAhyWQvkKEgXZhHI=
modernc.org/gc/v3 v3.0.0-20240107210532-573471604cb6/go.mod h1:Qz0X07sNOR1jWYCrJMEnbW/X55x206Q7Vt4mz6/wHp4=
modernc.org/libc v1.41.0 h1:g9YAc6BkKlgORsUWj+JwqoB1wU3o4DE3bM3yvA3k+Gk=
modernc.org/libc v1.41.0/go.mod h1:w0eszPsiXoOnoMJgrXjglgLuDy/bt5RR4y3QzUUeodY=
modernc.org/mathutil v1.6.0 h1:fRe9+AmYlaej+64JsEEhoWuAYBkOtQiMEU7n/XgfYi4=
modernc.org/mathutil v1.6.0/go.mod h1:Ui5Q9q1TR2gFm0AQRqQUaBWFLAhQpCwNcuhBOSedWPo=
modernc.org/memory v1.8.0 h1:IqGTL6eFMaDZZhEWwcREgeMXYwmW83LYW8cROZYkg+E=
modernc.org/memory v1.8.0/go.mod h1:XPZ936zp5OMKGWPqbD3JShgd/ZoQ7899TUuQqxY+peU=
modernc.org/sqlite v1.29.5 h1:8l/SQKAjDtZFo9lkJLdk8g9JEOeYRG4/ghStDCCTiTE=
modernc.org/sqlite v1.29.5/go.mod h1:S02dvcmm7TnTRvGhv8IGYyLnIt7AS2KPaB1F/71p75U=
modernc.org/strutil v1.2.0 h1:agBi9dp1I+eOnxXeiZawM8F4LawKv4NzGWSaLfyeNZA=
modernc.org/strutil v1.2.0/go.mod h1:/mdcBmfOibveCTBxUl5B5l6W+TTH1FXPLHZE6bTosX0=
modernc.org/token v1.1.0 h1:Xl7Ap9dKaEs5kLoOQeQmPWevfnk/DM5qcLcYlA8ys6Y=
modernc.org/token v1.1.0/go.mod h1:UGzOrNV1mAFSEB63lOFHIpNRUVMvYTc6yu1SMY/XTDM=



// greener/html.go

// Package greener provides more efficient ways of building web applications
package greener

import (
	"fmt"
	"html/template"
	"strings"
)

// Text escapes some specical characters and returns a template.HTML which the html/template package will treat as HTML without further escaping.
func Text(t string) template.HTML {
	return template.HTML(template.HTMLEscapeString(t))
}

// HTMLPrintf takes a string containing %s characters and a set of template.HTML strings and returns an template.HTML with the placeholders substituted. This is faster than using template/html Template objects by about 8x but less safe in that no context specific checks about where you are substituing things are made.
func HTMLPrintf(h string, hws ...template.HTML) template.HTML {
	hs := []interface{}{}
	for _, hw := range hws {
		hs = append(hs, hw)
	}
	return template.HTML(fmt.Sprintf(h, hs...))
}

func ConcatenateHTML(htmlParts []template.HTML, separator string) template.HTML {
	var parts []string
	for _, part := range htmlParts {
		parts = append(parts, string(part))
	}
	return template.HTML(strings.Join(parts, separator))
}

// HTMLBuilder wraps strings.Builder to specifically handle template.HTML.
type HTMLBuilder struct {
	builder strings.Builder
}

// WriteHTML appends the string representation of template.HTML to the builder.
func (hb *HTMLBuilder) WriteHTML(html template.HTML) {
	hb.builder.WriteString(string(html))
}

// String returns the accumulated strings as a template.HTML.
func (hb *HTMLBuilder) HTML() template.HTML {
	return template.HTML(hb.builder.String())
}

func (hb *HTMLBuilder) Printf(h string, hws ...template.HTML) {
	hs := []interface{}{}
	for _, hw := range hws {
		hs = append(hs, hw)
	}
	hb.builder.WriteString(fmt.Sprintf(h, hs...))
}



// greener/icons.go

package greener

import (
	"bytes"
	"fmt"
	"html/template"
	"image"
	"image/png"
	"io/fs"
	"net/url"
	"sync"

	"golang.org/x/image/draw"
)

// ImageData holds the resized image and its ETag
type ImageData struct {
	Image image.Image
	Size  string
}

func resizeImage(src image.Image, size int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.NearestNeighbor.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)
	return dst
}

func loadAndResizeImage(icon512Bytes []byte, icon512 image.Image, sizes []int) (map[int][]byte, error) {
	resizedImages := make(map[int][]byte)
	resizedImages[512] = icon512Bytes

	var wg sync.WaitGroup
	var mutex sync.Mutex                    // Mutex to protect map writes
	errChan := make(chan error, len(sizes)) // Buffered channel to store errors from goroutines

	for _, size := range sizes {
		wg.Add(1)
		go func(size int) {
			defer wg.Done()
			resized := resizeImage(icon512, size)
			buf := new(bytes.Buffer)
			if err := png.Encode(buf, resized); err != nil {
				errChan <- err
				return
			}
			mutex.Lock() // Lock the mutex before accessing the shared map
			resizedImages[size] = buf.Bytes()
			mutex.Unlock() // Unlock the mutex after the map is updated
		}(size)
	}

	wg.Wait()
	close(errChan) // Close the channel to signal no more values will be sent

	// Check if there were any errors
	for err := range errChan {
		if err != nil {
			return nil, err // Return the first error encountered
		}
	}

	return resizedImages, nil
}

// func StaticIconHandler(logger Logger, icon512 image.Image, etag string, sizes []int) http.HandlerFunc {
// 	resizedImages, err := loadAndResizeImage(icon512, etag, sizes)
// 	if err != nil {
// 		log.Fatalf("Failed to resize icons: %v\n", err)
// 	}
//
// 	return func(w http.ResponseWriter, r *http.Request) {
// 		// Determine requested size from URL
// 		path := r.URL.Path
// 		var requestedSize int
// 		var sizeFound bool
//
// 		for size := range resizedImages {
// 			if strings.Contains(path, strconv.Itoa(size)+"x"+strconv.Itoa(size)) {
// 				requestedSize = size
// 				sizeFound = true
// 				break
// 			}
// 		}
//
// 		if !sizeFound {
// 			http.NotFound(w, r)
// 			return
// 		}
//
// 		// Check for ETag match to possibly return 304 Not Modified
// 		// logger.Logf("ETag: %s, %s", r.Header.Get("If-None-Match"), resizedImages[requestedSize].ETag)
// 		if r.Header.Get("If-None-Match") == resizedImages[requestedSize].ETag {
// 			w.WriteHeader(http.StatusNotModified)
// 			return
// 		}
//
// 		// Serve the requested size
// 		w.Header().Set("Content-Type", "image/png")
// 		w.Header().Set("ETag", resizedImages[requestedSize].ETag)
// 		err := png.Encode(w, resizedImages[requestedSize].Image)
// 		if err != nil {
// 			logger.Logf("Failed to encode image: %v", err)
// 			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
// 		}
// 	}
// }

type DefaultIconsInjector struct {
	Logger
	icon512      image.Image
	sizes        []int
	cacheSeconds int
	html         string
	paths        map[int]string
	chs          map[int]ContentHandler
}

func (d *DefaultIconsInjector) IconPaths() map[int]string {
	return d.paths
}

func (d *DefaultIconsInjector) Inject(mux HandlerRouter) (template.HTML, template.HTML) {
	for size := range d.chs {
		mux.Handle("/"+d.paths[size], d.chs[size])
	}
	return template.HTML(d.html), template.HTML("")
}

func NewDefaultIconsInjector(logger Logger, iconFS fs.FS, icon512Path string, sizes []int, cacheSeconds int) (*DefaultIconsInjector, error) {
	icon512Bytes, err := fs.ReadFile(iconFS, icon512Path)
	if err != nil {
		logger.Logf("Failed to open source image for favicon: %v", err)
		return nil, err
	}
	icon512, _, err := image.Decode(bytes.NewReader(icon512Bytes))
	if err != nil {
		logger.Logf("Failed to decode source image for favicon: %v", err)
		return nil, err
	}

	d := &DefaultIconsInjector{
		Logger:       logger,
		icon512:      icon512,
		sizes:        sizes,
		cacheSeconds: cacheSeconds,
		paths:        make(map[int]string),
		chs:          make(map[int]ContentHandler),
		html:         "",
	}
	d.Logf("Injecting route and HTML for png icons")
	resizedImages, err := loadAndResizeImage(icon512Bytes, icon512, d.sizes)
	if err != nil {
		logger.Logf("Failed to resize icons: %v\n", err)
		return nil, err
	}
	for _, size := range sizes {
		imageData := resizedImages[size]
		d.chs[size] = NewContentHandler(d.Logger, imageData, "image/png", "", d.cacheSeconds)
		path := fmt.Sprintf("icon-%dx%d-%s.png", size, size, d.chs[size].Hash())
		encodedPath := url.PathEscape(path)
		d.html += fmt.Sprintf(\`
    <link rel="icon" type="image/png" sizes="%dx%d" href="/%s">\`, size, size, encodedPath)
		if size == 180 {
			d.html += fmt.Sprintf(\`
    <link rel="apple-touch-icon" sizes="180x180" href="/%s">\`, encodedPath)
		}
		d.paths[size] = path
	}
	return d, nil
}



// greener/kv.go

// rm kvstore.db; go run cmd/kvstore/main.go
// Note: If you make create/drop tables outside of this code, it won't notice until you restart. You therefore shouldn't do that.
// Also an application should only have one KV, otherwise the mutex won't behave correclty and you may get database is locked errors due to multiple writers.

package greener

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// JSONValue is a map that can only contain number or string values but will be stored encoded in JSON.
type JSONValue map[string]interface{}

// MarshalJSON ensures the values are either float64 or string.
func (j JSONValue) MarshalJSON() ([]byte, error) {
	temp := make(map[string]interface{})
	for k, v := range j {
		switch v.(type) {
		case float64, string:
			temp[k] = v
		default:
			return nil, fmt.Errorf("JSONValue must be a map of strings to either float64 or string values")
		}
	}
	return json.Marshal(temp)
}

// UnmarshalJSON ensures the values are either float64 or string.
func (j *JSONValue) UnmarshalJSON(data []byte) error {
	temp := make(map[string]interface{})
	if err := json.Unmarshal(data, &temp); err != nil {
		return err
	}

	for _, v := range temp {
		switch v.(type) {
		case float64, string:
			continue
		default:
			return fmt.Errorf("JSONValue must be a map of strings to either float64 or string values")
		}
	}
	*j = temp
	return nil
}

// Row represents a single row returned by the Iterate method.
type Row struct {
	PK      string
	SK      string
	Expires *time.Time // This is a pointer so that it can be nil, representing a NULL value in SQL
	Data    JSONValue
}

// KvStore is the interface defining the key value store operations.
type KvStore interface {
	Create(pk string, sk string, data JSONValue, expires *time.Time) error
	Put(pk string, sk string, data JSONValue, expires *time.Time) error
	Delete(pk, sk string) error
	Iterate(pk, skStart string, limit int, after string) ([]Row, string, error)
	Get(pk, sk string) (JSONValue, *time.Time, error)
}

// KV keeps track of KVstore tables and manages the database connection.
type KV struct {
	db DB
}

// NewKV initializes and returns a new KV.
func NewKV(ctx context.Context, db DB) (*KV, error) {
	tm := &KV{
		db: db,
	}
	err := tm.db.Write(func(writeDB WriteDBHandler) error {
		tableName := "kv"
		createTableSQL := fmt.Sprintf(\`
		CREATE TABLE IF NOT EXISTS %s (
		    pk TEXT NOT NULL,
		    sk TEXT NOT NULL,
		    data JSON NOT NULL,
		    expires INTEGER,
		    PRIMARY KEY (pk, sk)
		);\`, tableName)
		_, err := writeDB.ExecContext(ctx, createTableSQL)
		if err != nil {
			return fmt.Errorf("failed to create table %s: %w", tableName, err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return tm, nil
}

// StartCleanupRoutine runs a goroutine that periodically deletes expired rows from KVstore tables.
func (tm *KV) StartCleanupRoutine(ctx context.Context) {
	ticker := time.NewTicker(60 * time.Second)
	go func() {
		for {
			select {
			case <-ticker.C:
				now := time.Now().Unix()
				tableName := "kv"
				err := tm.db.Write(func(writeDB WriteDBHandler) error {
					_, err := writeDB.ExecContext(ctx, "DELETE FROM "+tableName+" WHERE expires IS NOT NULL AND expires < ?", now)
					return err
				})
				if err != nil {
					log.Printf("Error cleaning up expired rows in table %s: %v", tableName, err)
				}
			}
		}
	}()
}

func (tm *KV) putOrCreate(ctx context.Context, pk string, sk string, data JSONValue, expires *time.Time, allowUpdate bool) error {

	tableName := "kv"
	changed := true
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("error encoding data to JSON: %w", err)
	}
	var expiresUnix *int64
	if expires != nil {
		unix := expires.Unix()
		expiresUnix = &unix
	}
	err = tm.db.Write(func(writeDB WriteDBHandler) error {
		if allowUpdate {
			upsertSQL := fmt.Sprintf(\`
        	    INSERT INTO %s (pk, sk, data, expires) VALUES (?, ?, ?, ?)
        	    ON CONFLICT(pk, sk) DO UPDATE SET data=excluded.data, expires=excluded.expires;
        	\`, tableName)
			_, err = writeDB.ExecContext(ctx, upsertSQL, pk, sk, jsonData, expiresUnix)
			if err != nil {
				return fmt.Errorf("failed to upsert row in table %s: %w", tableName, err)
			}
			return nil
		} else {
			insertSQL := fmt.Sprintf(\`
        	    INSERT INTO %s (pk, sk, data, expires) VALUES (?, ?, ?, ?)
        	    ON CONFLICT(pk, sk) DO NOTHING;
        	\`, tableName)
			result, err := writeDB.ExecContext(ctx, insertSQL, pk, sk, jsonData, expiresUnix)
			if err != nil {
				return fmt.Errorf("failed to insert row in table %s: %w", tableName, err)
			}
			rowsAffected, err := result.RowsAffected()
			if err != nil {
				return fmt.Errorf("error checking rows affected for table %s: %w", tableName, err)
			}
			if rowsAffected == 0 {
				// Row with pk and sk already exists and we have simply ignored it.
				// Crucially, this is not an error
				changed = false
			}
			return nil
		}
	})
	if err != nil {
		return fmt.Errorf("failed to put row in table %s: %w", tableName, err)
	}
	if !allowUpdate && !changed {
		// The create failed
		return fmt.Errorf("row with pk %s and sk %s already exists", pk, sk)
	}
	return nil
}

// Put inserts or updates a row with the given pk, sk, data, and expires.
func (tm *KV) Put(ctx context.Context, pk string, sk string, data JSONValue, expires *time.Time) error {
	return tm.putOrCreate(ctx, pk, sk, data, expires, true) // true allows updates
}

// Create inserts a row with the given pk, sk, data, and expires, but fails if the row already exists.
func (tm *KV) Create(ctx context.Context, pk string, sk string, data JSONValue, expires *time.Time) error {
	return tm.putOrCreate(ctx, pk, sk, data, expires, false) // false disallows updates, failing on conflict
}

// Get retrieves a row with the given pk and sk. It returns the data and expires if the row exists and is not expired.
func (tm *KV) Get(ctx context.Context, pk string, sk string) (JSONValue, *time.Time, error) {
	tableName := "kv"

	querySQL := fmt.Sprintf(\`
        SELECT data, expires FROM %s WHERE pk = ? AND sk = ? AND (expires IS NULL OR expires > ?);
    \`, tableName)

	var jsonData string
	var expiresUnix sql.NullInt64
	err := tm.db.QueryRowContext(ctx, querySQL, pk, sk, time.Now().Unix()).Scan(&jsonData, &expiresUnix)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil, fmt.Errorf("no matching row found")
		}
		return nil, nil, fmt.Errorf("error querying for row: %w", err)
	}

	var data JSONValue
	if err := json.Unmarshal([]byte(jsonData), &data); err != nil {
		return nil, nil, fmt.Errorf("error decoding data from JSON: %w", err)
	}

	var expires *time.Time
	if expiresUnix.Valid {
		t := time.Unix(expiresUnix.Int64, 0)
		expires = &t
	}

	return data, expires, nil
}

// Delete removes a row with the given pk and sk from the table.
func (tm *KV) Delete(ctx context.Context, pk string, sk string) error {
	tableName := "kv"

	// Prepare the DELETE statement
	deleteSQL := fmt.Sprintf("DELETE FROM %s WHERE pk = ? AND sk = ?", tableName)

	err := tm.db.Write(func(writeDB WriteDBHandler) error {
		_, err := writeDB.ExecContext(ctx, deleteSQL, pk, sk)
		if err != nil {
			return fmt.Errorf("failed to delete row from table %s with pk %s and sk %s: %w", tableName, pk, sk, err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to delete row from table %s with pk %s and sk %s: %w", tableName, pk, sk, err)
	}
	return err
}

// Iterate over rows in a table based on primary key and sort key.
// If 'after' is true, search for rows with sort keys strictly greater than 'sk'.
// Otherwise, include rows with sort keys greater than or equal to 'sk'.
func (tm *KV) Iterate(ctx context.Context, pk, sk string, limit int, after bool) ([]Row, string, error) {
	tableName := "kv"

	var rows []Row
	var querySQL string
	var args []interface{}

	skCondition := ">="
	if after {
		skCondition = ">"
	}

	if sk != "" {
		querySQL = fmt.Sprintf(\`
            SELECT pk, sk, data, expires FROM %s
            WHERE pk = ? AND sk %s ? AND (expires IS NULL OR expires > ?)
            ORDER BY sk ASC
            LIMIT ?;\`, tableName, skCondition)
		args = []interface{}{pk, sk, time.Now().Unix(), limit}
	} else {
		querySQL = fmt.Sprintf(\`
            SELECT pk, sk, data, expires FROM %s
            WHERE pk = ? AND (expires IS NULL OR expires > ?)
            ORDER BY sk ASC
            LIMIT ?;\`, tableName)
		args = []interface{}{pk, time.Now().Unix(), limit}
	}

	sqlRows, err := tm.db.QueryContext(ctx, querySQL, args...)
	if err != nil {
		return nil, "", fmt.Errorf("error executing iterate query: %w", err)
	}
	defer sqlRows.Close()

	for sqlRows.Next() {
		var r Row
		var expiresUnix sql.NullInt64
		var jsonData string
		if err := sqlRows.Scan(&r.PK, &r.SK, &jsonData, &expiresUnix); err != nil {
			return nil, "", fmt.Errorf("error scanning row: %w", err)
		}

		if err := json.Unmarshal([]byte(jsonData), &r.Data); err != nil {
			return nil, "", fmt.Errorf("error unmarshaling JSON data: %w", err)
		}

		if expiresUnix.Valid {
			expires := time.Unix(expiresUnix.Int64, 0)
			r.Expires = &expires
		}

		rows = append(rows, r)
	}

	// Generate a new 'after' token for pagination, based on the last 'sk' value seen
	newAfter := sk
	if len(rows) > 0 {
		newAfter = rows[len(rows)-1].SK
	}

	return rows, newAfter, nil
}



// greener/serve.go

package greener

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

type Logger interface {
	Logf(string, ...interface{})
	// Errorf(string, ...interface{})
}

type DefaultLogger struct {
	logf func(string, ...interface{})
}

func (cl *DefaultLogger) Logf(m string, a ...interface{}) {
	cl.logf(m, a...)
}

// func (cl *DefaultLogger) Errorf(m string, a ...interface{}) {
// 	cl.logf("ERROR: "+m, a...)
// }

func NewDefaultLogger(logf func(string, ...interface{})) *DefaultLogger {
	return &DefaultLogger{logf: logf}
}

func AutoServe(logger Logger, mux *http.ServeMux) (error, context.Context, func()) {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	portStr := os.Getenv("PORT")
	if portStr == "" {
		portStr = "8000"
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return err, nil, nil
	}
	origin := os.Getenv("ORIGIN")
	host := os.Getenv("HOST")
	if host == "" {
		host = "localhost"
	}
	uds := os.Getenv("UDS")
	mux.HandleFunc("/livez", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})
	go Serve(ctx, logger, mux, host, port, uds)
	healthURL := ""
	if origin != "" {
		healthURL = fmt.Sprintf("%s/livez", origin)
	} else if uds == "" {
		healthURL = fmt.Sprintf("http://%s:%d/livez", host, port)
	}
	if healthURL != "" {
		if err = PollForHealth(healthURL, 2*time.Second, 20*time.Millisecond); err != nil {
			return err, nil, nil
		}
	}
	return nil, ctx, stop
}

func Serve(ctx context.Context, logger Logger, handler http.Handler, host string, port int, uds string) {
	addr := fmt.Sprintf("%s:%d", host, port)
	srv := &http.Server{
		Addr:    addr,
		Handler: handler,
	}
	if uds != "" {
		listener, err := net.Listen("unix", uds)
		if err != nil {
			logger.Logf("Error listening: %v", err)
			return
		}
		logger.Logf("Server listening on Unix Domain Socket: %s", uds)
		if err := srv.Serve(listener); err != http.ErrServerClosed {
			logger.Logf("Server closed with error: %v", err)
			return
		}
	} else {
		logger.Logf("Server listening on %s", addr)
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			logger.Logf("Server closed with error: %v", err)
			return
		}
	}
}



// greener/sqlitec.go

//go:build sqlitec
// +build sqlitec

package greener

import (
	_ "github.com/mattn/go-sqlite3"
)

var SqlDriver = "sqlite3"
var sqlitePragmas = map[string]string{
	"_journal_mode": "WAL",
	"_busy_timeout": "5000",
	"_synchronous":  "NORMAL",
	"_cache_size":   "1000000000", // 1GB
	"_foreign_keys": "true",
	"temp_store":    "memory",
	"_txlock":       "immediate",
	// "cache":         "shared",
}




// greener/sqlitego.go

//go:build !sqlitec
// +build !sqlitec

package greener

import (
	_ "modernc.org/sqlite"
)

var SqlDriver = "sqlite"
var sqlitePragmas = map[string]string{
	"_journal_mode": "WAL",
	"_busy_timeout": "5000",
	"_synchronous":  "NORMAL",
	"_cache_size":   "1000000000", // 1GB
	"_foreign_keys": "true",
	"temp_store":    "memory",
	"_txlock":       "immediate",
	"cache":         "shared",
}



// greener/tx.go

package greener

import (
	"context"
	"database/sql"
	"fmt"
)

type txWrapper struct {
	tx  *sql.Tx
	err error
}

func (t *txWrapper) Abort(err error) {
	if t.err != nil {
		panic("Abort called again when there was already an error")
	}
	t.err = err
	rollbackErr := t.tx.Rollback()
	if rollbackErr != nil {
		fmt.Printf("Error rolling back: %v. Original error: %v\n", rollbackErr, err)
	}
}

func (t *txWrapper) ExecContext(ctx context.Context, query string, args ...interface{}) (*resultWrapper, error) {
	if t.err != nil {
		return nil, fmt.Errorf("this transaction is already aborted")
	}
	result, err := t.tx.ExecContext(ctx, query, args...)
	if err != nil {
		t.Abort(err)
		return nil, err
	}
	return &resultWrapper{result: result, txWrapper: t}, nil
}

type resultWrapper struct {
	result    sql.Result
	txWrapper *txWrapper
}

func (rw *resultWrapper) LastInsertId() (int64, error) {
	if rw.txWrapper.err != nil {
		return 0, fmt.Errorf("this transaction is already aborted")
	}
	id, err := rw.result.LastInsertId()
	if err != nil {
		rw.txWrapper.Abort(err)
	}
	return id, err
}

func (rw *resultWrapper) RowsAffected() (int64, error) {
	if rw.txWrapper.err != nil {
		return 0, fmt.Errorf("this transaction is already aborted")
	}
	count, err := rw.result.RowsAffected()
	if err != nil {
		rw.txWrapper.Abort(err)
	}
	return count, err
}

func (t *txWrapper) QueryContext(ctx context.Context, query string, args ...interface{}) (*rowsWrapper, error) {
	if t.err != nil {
		return nil, fmt.Errorf("this transaction is already aborted")
	}
	rows, err := t.tx.QueryContext(ctx, query, args...)
	if err != nil {
		t.Abort(err)
		return nil, err
	}
	return &rowsWrapper{rows: rows, txWrapper: t}, nil
}

func (t *txWrapper) QueryRowContext(ctx context.Context, query string, args ...interface{}) *rowWrapper {
	if t.err != nil {
		return &rowWrapper{row: nil, txWrapper: t}
	}
	return &rowWrapper{row: t.tx.QueryRowContext(ctx, query, args...), txWrapper: t}
}

type rowWrapper struct {
	row       *sql.Row
	txWrapper *txWrapper
}

func (r *rowWrapper) Scan(dest ...interface{}) error {
	if r.txWrapper.err != nil {
		return fmt.Errorf("this transaction is already aborted")
	}
	err := r.row.Scan(dest...)
	if err != nil {
		r.txWrapper.Abort(err)
	}
	return err
}

type rowsWrapper struct {
	rows      *sql.Rows
	txWrapper *txWrapper
}

func (r *rowsWrapper) Scan(dest ...interface{}) error {
	if r.txWrapper.err != nil {
		return fmt.Errorf("this transaction is already aborted")
	}
	err := r.rows.Scan(dest...)
	if err != nil {
		r.txWrapper.Abort(err)
	}
	return err
}

func (r *rowsWrapper) Next() bool {
	if r.txWrapper.err != nil {
		return false
	}
	return r.rows.Next()
}

func (r *rowsWrapper) Close() error {
	if r.txWrapper.err != nil {
		return fmt.Errorf("this transaction is already aborted")
	}
	return r.rows.Close()
}

func (r *rowsWrapper) Err() error {
	if r.txWrapper.err != nil {
		return fmt.Errorf("this transaction is already aborted")
	}
	return r.rows.Err()
}



// greener/ui.go

package greener

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"strings"
)

// UISupport
type StyleProvider interface {
	Style() string
}

type ScriptProvider interface {
	Script() string
}

type ServiceWorkerProvider interface {
	ServiceWorker() string
}

type UISupport interface {
	StyleProvider
	ScriptProvider
	ServiceWorkerProvider
}

// Injector
type Injector interface {
	Inject(HandlerRouter) (template.HTML, template.HTML)
}

type HandlerRouter interface {
	Handle(string, http.Handler)
}

// EmptyPageProvider
type EmptyPageProvider interface {
	PerformInjections(HandlerRouter)
	Page(title string, body template.HTML) template.HTML
}

// DefaultStyleProvider implements StyleProvider
type DefaultStyleProvider struct {
	style string
}

func (dsp *DefaultStyleProvider) Style() string {
	return dsp.style
}

// DefaultScriptProvider implements ScriptProvider
type DefaultScriptProvider struct {
	script string
}

func (dsp *DefaultScriptProvider) Script() string {
	return dsp.script
}

// DefaultServiceWorkerProvider implements ServiceWorkerProvider
type DefaultServiceWorkerProvider struct {
	serviceWorker string
}

func (dsp *DefaultServiceWorkerProvider) ServiceWorker() string {
	return dsp.serviceWorker
}

// DefaultUISupport implements UISupport by embedding StyleProvider ScriptProvider and ServiceWorkerProvider
type DefaultUISupport struct {
	StyleProvider
	ScriptProvider
	ServiceWorkerProvider
}

// NewDefaultUISupport creates a DefaultUISupport from strings representing the style, the script and the serviceworker fragments for the component. Each can be "" to indicate the conponent doesn't need them.
func NewDefaultUISupport(style, script, serviceWorker string) *DefaultUISupport {
	return &DefaultUISupport{
		StyleProvider:         &DefaultStyleProvider{style: style},
		ScriptProvider:        &DefaultScriptProvider{script: script},
		ServiceWorkerProvider: &DefaultServiceWorkerProvider{serviceWorker: serviceWorker},
	}
}

type DefaultStyleInjector struct {
	Logger
	uiSupports   []UISupport
	cacheSeconds int
}

func (d *DefaultStyleInjector) Inject(mux HandlerRouter) (template.HTML, template.HTML) {
	var buffer bytes.Buffer
	for _, uis := range d.uiSupports {
		buffer.WriteString(uis.Style())
	}
	style := buffer.Bytes()
	if style != nil {
		d.Logf("Injecting route and HTML for styles")
		ch := NewContentHandler(d.Logger, style, "text/css", "", d.cacheSeconds)
		mux.Handle("/style-"+ch.Hash()+".css", ch)
		return HTMLPrintf(\`
    <link rel="stylesheet" href="/style-%s.css">\`, Text(url.PathEscape(ch.Hash()))), template.HTML("")
	} else {
		d.Logf("No styles specified")
		return template.HTML(""), template.HTML("")
	}
}
func NewDefaultStyleInjector(logger Logger, uiSupports []UISupport, cacheSeconds int) *DefaultStyleInjector {
	return &DefaultStyleInjector{Logger: logger, uiSupports: uiSupports, cacheSeconds: cacheSeconds}
}

type DefaultScriptInjector struct {
	Logger
	uiSupports   []UISupport
	cacheSeconds int
}

func (d *DefaultScriptInjector) Inject(mux HandlerRouter) (template.HTML, template.HTML) {
	// Handle service worker first
	var swBuffer bytes.Buffer
	for _, sp := range d.uiSupports {
		swBuffer.WriteString(sp.ServiceWorker())
	}
	serviceWorker := swBuffer.Bytes()
	if serviceWorker != nil {
		d.Logf("Injecting route for /service-worker.js")
		// No cache for this one
		ch := NewContentHandler(d.Logger, serviceWorker, "text/javascript; charset=utf-8", "", 0)
		mux.Handle("/service-worker.js", ch)
	} else {
		d.Logf("No service workers specified")
	}

	var buffer bytes.Buffer
	for _, uis := range d.uiSupports {
		buffer.WriteString(uis.Script())
	}
	if serviceWorker != nil {
		buffer.WriteString(\`
/* Service Worker */

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js').then(registration => {
      console.log('ServiceWorker registration successful with scope: ', registration.scope);
    }, err => {
      console.log('ServiceWorker registration failed: ', err);
    });
  });
}
\`)
	}
	script := buffer.Bytes()
	if script != nil {
		d.Logf("Injecting route and HTML for script")
		ch := NewContentHandler(d.Logger, script, "text/javascript; charset=utf-8", "", d.cacheSeconds)
		mux.Handle("/script-"+ch.Hash()+".js", ch)
		return template.HTML(""), HTMLPrintf(\`
    <script src="/script-%s.js"></script>\`, Text(url.PathEscape(ch.Hash())))
	} else {
		d.Logf("No scripts specified")
		return template.HTML(""), template.HTML("")
	}
}

func NewDefaultScriptInjector(logger Logger, uiSupports []UISupport, cacheSeconds int) *DefaultScriptInjector {
	return &DefaultScriptInjector{Logger: logger, uiSupports: uiSupports, cacheSeconds: cacheSeconds}
}

type DefaultThemeColorInjector struct {
	Logger
	themeColor string
}

func (d *DefaultThemeColorInjector) Inject(mux HandlerRouter) (template.HTML, template.HTML) {
	d.Logf("Injecting HTML for theme color")
	return HTMLPrintf(\`
    <meta name="msapplication-TileColor" content="%s">
    <meta name="theme-color" content="%s">\`, Text(d.themeColor), Text(d.themeColor)), template.HTML("")
}

func NewDefaultThemeColorInjector(logger Logger, themeColor string) *DefaultThemeColorInjector {
	return &DefaultThemeColorInjector{Logger: logger, themeColor: themeColor}
}

type DefaultSEOInjector struct {
	Logger
	description string
}

func (d *DefaultSEOInjector) Inject(mux HandlerRouter) (template.HTML, template.HTML) {
	d.Logf("Adding HTML for SEO meta description")
	return HTMLPrintf(\`
    <meta name="description" content="%s">\`, Text(d.description)), template.HTML("")
}

func NewDefaultSEOInjector(logger Logger, description string) *DefaultSEOInjector {
	return &DefaultSEOInjector{Logger: logger, description: description}
}

type DefaultManifestInjector struct {
	Logger
	appShortName string
	themeColor   string
	cacheSeconds int
	startURL     string
	icons        []icon
}

type icon struct {
	Src   string \`json:"src"\`
	Sizes string \`json:"sizes"\`
	Type  string \`json:"type"\`
}

func (d *DefaultManifestInjector) Inject(mux HandlerRouter) (template.HTML, template.HTML) {
	manifestData := struct {
		Name       string \`json:"name"\`
		ShortName  string \`json:"short_name"\`
		StartURL   string \`json:"start_url"\`
		Display    string \`json:"display"\`
		ThemeColor string \`json:"theme_color"\`
		Icons      []icon \`json:"icons"\`
	}{
		Name:       d.appShortName,
		ShortName:  d.appShortName,
		StartURL:   d.startURL,
		Display:    "standalone",
		ThemeColor: d.themeColor,
		Icons:      d.icons,
	}
	manifest, err := json.MarshalIndent(manifestData, "", "  ")
	if err != nil {
		d.Logf("JSON marshalling failed: %s", err)
		panic("Could not generate JSON for the manifest. Perhaps a problem with the config?")
	}
	d.Logf("Adding route for manifest")
	ch := NewContentHandler(d.Logger, manifest, "application/json", "", d.cacheSeconds)
	mux.Handle("/manifest.json", ch)
	return template.HTML(\`
    <link rel="manifest" href="/manifest.json">\`), template.HTML("")
}

func NewDefaultManifestInjector(logger Logger, appShortName string, themeColor string, startURL string, cacheSeconds int, iconPaths map[int]string, sizes []int) (*DefaultManifestInjector, error) {
	var icons []icon

	for _, size := range sizes {
		path, exists := iconPaths[size]
		if !exists {
			// Handle the case where no path is found for a given size
			return nil, fmt.Errorf("no path found for size: %d", size)
		}

		icons = append(icons, icon{
			Src:   "/" + path,
			Sizes: fmt.Sprintf("%dx%d", size, size),
			Type:  "image/png",
		})
	}

	return &DefaultManifestInjector{Logger: logger, appShortName: appShortName, themeColor: themeColor, cacheSeconds: cacheSeconds, icons: icons, startURL: startURL}, nil

}

// Injectors prepares an HTML page string (to be used with HTMLPrintf) from a slice of Injector.
type DefaultEmptyPageProvider struct {
	page      string
	injectors []Injector
}

func (d *DefaultEmptyPageProvider) Page(title string, body template.HTML) template.HTML {
	return HTMLPrintf(d.page, Text(title), body)
}

func (d *DefaultEmptyPageProvider) PerformInjections(mux HandlerRouter) {
	headExtra := ""
	bodyExtra := ""
	for _, injector := range d.injectors {
		h, b := injector.Inject(mux)
		headExtra += strings.Replace(string(h), "%", "%%", -1)
		bodyExtra += strings.Replace(string(b), "%", "%%", -1)
	}
	d.page = \`<!DOCTYPE html>
<html lang="en-GB">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%s</title>\` + headExtra + \`
  </head>
  <body>
%s\` + bodyExtra + \`
  </body>
</html>\`
}

func NewDefaultEmptyPageProvider(injectors []Injector) *DefaultEmptyPageProvider {
	return &DefaultEmptyPageProvider{injectors: injectors}
}



// greener/www.go

package greener

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"strings"
)

// ETagEntry represents an entry in the etags.json file
type ETagEntry struct {
	MTime string \`json:"mtime"\`
	ETag  string \`json:"etag"\`
}

// CompressedFileHandler is a handler that serves compressed files.
type CompressedFileHandler struct {
	wwwHandler   http.Handler
	wwwgzHandler http.Handler
	wwwFS        fs.FS
	wwwgzFS      fs.FS
	etagMap      map[string]string
}

// NewCompressedFileHandler creates a new CompressedFileHandler.
func NewCompressedFileHandler(wwwFS, wwwgzFS fs.FS, etagMap map[string]string) *CompressedFileHandler {
	return &CompressedFileHandler{
		wwwHandler:   http.FileServer(http.FS(wwwFS)),
		wwwgzHandler: http.FileServer(http.FS(wwwgzFS)),
		wwwFS:        wwwFS,
		wwwgzFS:      wwwgzFS,
		etagMap:      etagMap,
	}
}

// LoadEtagsJSON parses the etags.json file and creates a map of paths to ETags.
func LoadEtagsJSON(data []byte) (map[string]string, error) {
	var etagFile struct {
		Entries map[string]ETagEntry \`json:"entries"\`
	}
	err := json.Unmarshal(data, &etagFile)
	if err != nil {
		return nil, err
	}

	etagMap := make(map[string]string)
	for path, entry := range etagFile.Entries {
		etagMap[path] = entry.ETag
	}

	return etagMap, nil
}

// ServeHTTP serves HTTP requests, checking for compressed versions of files.
func (h *CompressedFileHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	requestPath := r.URL.Path[1:]
	if etag, ok := h.etagMap[requestPath]; ok {
		etag = fmt.Sprintf("\"%s\"", etag)
		w.Header().Set("ETag", etag)
		if match := r.Header.Get("If-None-Match"); match != "" {
			if match = strings.TrimSpace(match); EtagMatch(match, etag) {
				w.WriteHeader(http.StatusNotModified)
				return
			}
		}
	}

	// Parse Accept-Encoding header
	acceptEncoding := r.Header.Get("Accept-Encoding")
	encodings := ParseEncodings(acceptEncoding)

	// Determine if an encoding is acceptable based on q-values
	isAcceptable := func(encoding string) bool {
		q, exists := encodings[encoding]
		return exists && q > 0
	}

	// Check if gzip is acceptable and the file exists in the wwwgz filesystem
	if isAcceptable("gzip") {
		gzipStat, err := fs.Stat(h.wwwgzFS, requestPath)
		if err == nil && gzipStat.Size() > 0 {
			w.Header().Set("Content-Encoding", "gzip")
			h.wwwgzHandler.ServeHTTP(w, r)
			return
		}
	}

	// Serve the original request from the www filesystem
	h.wwwHandler.ServeHTTP(w, r)
}

`