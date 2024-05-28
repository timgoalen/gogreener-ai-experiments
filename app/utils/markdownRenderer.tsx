import Markdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/default.css";

interface Props {
    children: string;
  }

export default function MarkdownWithSyntaxHighlight({ children }: Props) {
  return <Markdown rehypePlugins={[rehypeHighlight]}>{children}</Markdown>;
}
