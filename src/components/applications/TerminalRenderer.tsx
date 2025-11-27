// TerminalRenderer.tsx
import { Renderer, Tokens } from 'marked';

class TerminalRenderer extends Renderer {
    heading({ text, depth, tokens }: Tokens.Heading): string {
        const colors: { [key: number]: string } = {
            1: '\x1b[1;92m', // Bright green
            2: '\x1b[92m',   // Green
            3: '\x1b[96m',   // Cyan
            4: '\x1b[94m',   // Blue
            5: '\x1b[93m',   // Yellow
            6: '\x1b[95m',   // Magenta
        };
        const reset = '\x1b[0m';
        const prefix = '#'.repeat(depth);
        return `${colors[depth] || colors[1]}${prefix} ${text}${reset}\n`;
    }

    paragraph({ tokens }: Tokens.Paragraph): string {
        const text = this.parser.parseInline(tokens);
        return `${text}\n\n`;
    }

    strong({ tokens }: Tokens.Strong): string {
        const text = this.parser.parseInline(tokens);
        return `\x1b[1;92m${text}\x1b[0m`; // Bright green bold
    }

    em({ tokens }: Tokens.Em): string {
        const text = this.parser.parseInline(tokens);
        return `\x1b[3m${text}\x1b[0m`; // Italic
    }

    codespan({ text }: Tokens.Codespan): string {
        return `\x1b[96m${text}\x1b[0m`; // Cyan
    }

    code({ text }: Tokens.Code): string {
        return `\x1b[90m${text}\x1b[0m\n`; // Dim gray
    }

    blockquote({ tokens }: Tokens.Blockquote): string {
        const text = this.parser.parse(tokens);
        return `\x1b[90m│ ${text.replace(/\n/g, '\n│ ')}\x1b[0m\n`; // Gray quote
    }

    list(token: Tokens.List): string {
        const type = token.ordered ? '1.' : '•';
        return token.items.map(item => {
            const content = this.listitem(item);
            return `\x1b[92m${type}\x1b[0m ${content.trim()}`;
        }).join('\n') + '\n\n';
    }

    listitem(item: Tokens.ListItem): string {
        return this.parser.parse(item.tokens);
    }

    link({ href, title, tokens }: Tokens.Link): string {
        const text = this.parser.parseInline(tokens);
        return `\x1b[94m${text}\x1b[0m (\x1b[96m${href}\x1b[0m)`;
    }

    hr(token: Tokens.Hr): string {
        return '\x1b[90m' + '─'.repeat(50) + '\x1b[0m\n';
    }
}

export default TerminalRenderer;