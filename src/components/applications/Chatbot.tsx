import React, { useEffect, useRef, useState } from 'react';
import Window from '../os/Window';

export interface ChatbotProps extends WindowAppProps { }

// eslint-disable-next-line no-control-regex
const stripAnsi = (s: string) => s.replace(/\x1B\[[0-9;]*[A-Za-z]/g, '');

const ChatbotApp: React.FC<ChatbotProps> = (props) => {
    const [connected, setConnected] = useState(false);
    const [log, setLog] = useState<string[]>([]);
    const [input, setInput] = useState('');

    const wsRef = useRef<WebSocket | null>(null);
    const logRef = useRef<HTMLDivElement | null>(null);

    const appendLine = (line: string) => {
        setLog((prev) => [...prev, line]);
    };



    const appendChunk = React.useCallback((chunk: string) => {
        const text = stripAnsi(String(chunk));

        // Handle carriage return updates (spinner lines)
        if (text.includes('\r')) {
            const segments = text.split('\r');
            segments.forEach((seg: string) => {
                if (!seg) return;
                const lines = seg.split(/\r?\n/);
                const update = lines[0];
                // Replace the last line with the updated spinner text
                setLog((prev) => {
                    const next = [...prev];
                    if (next.length === 0) {
                        next.push(update);
                    } else {
                        const lastLine = next[next.length - 1];
                        // Don't overwrite user messages
                        if (lastLine.startsWith('> ')) {
                            next.push(update);
                        } else {
                            next[next.length - 1] = update;
                        }
                    }
                    // Any remaining lines after newline are normal appended logs
                    for (let i = 1; i < lines.length; i++) {
                        const l = lines[i];
                        if (l && l.length > 0) next.push(l);
                    }
                    return next;
                });
            });
            return;
        }

        // Normal newline-separated logs
        const lines = text.split(/\r?\n/).filter((l: string) => l.length > 0);
        if (lines.length) setLog((prev) => {
            const next = [...prev];
            lines.forEach((l: string) => {
                // Avoid duplicating identical consecutive lines
                if (next.length === 0 || next[next.length - 1] !== l) {
                    next.push(l);
                } else {
                    next[next.length - 1] = l; // collapse duplicate
                }
            });
            return next;
        });
    }, []);

    useEffect(() => {
        const connect = () => {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = process.env.NODE_ENV === 'development' ? 'localhost:4000' : window.location.host;
                const ws = new WebSocket(`${protocol}//${host}`);
                wsRef.current = ws;
                ws.onopen = () => {
                    setConnected(true);
                    appendLine('Connected to local chatbot bridge (npx akshit-sharma-cli)');
                };
                ws.onmessage = (ev) => {
                    appendChunk(ev.data as string);
                };
                ws.onerror = () => {
                    appendLine('WebSocket error. Retrying...');
                };
                ws.onclose = () => {
                    setConnected(false);
                    setTimeout(connect, 1000);
                };
            } catch (e) {
                setConnected(false);
                setTimeout(connect, 1000);
            }
        };
        connect();
        return () => {
            wsRef.current && wsRef.current.close();
        };
    }, [appendChunk]);

    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [log]);

    const send = () => {
        const ws = wsRef.current;
        const trimmed = input.trim();
        if (!trimmed) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            appendLine('Not connected. Unable to send.');
            return;
        }
        ws.send(trimmed);
        appendLine(`> ${trimmed}`);
        setInput('');
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') send();
    };

    return (
        <Window
            top={48}
            left={48}
            width={1100}
            height={800}
            windowTitle="Chatbot"
            windowBarIcon="windowExplorerIcon"
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
            bottomLeftText={'Powered by akshit-sharma-cli'}
        >
            <div className="site-page" style={styles.container}>
                <div style={styles.header}>
                    <h2 style={window.innerWidth <= 768 ? { fontSize: '18px' } : {}}>AKSHIT'S ENHANCED AI ASSISTANT</h2>
                    <p style={styles.status}>{connected ? '● Connected' : '○ Connecting...'}</p>
                </div>
                <div style={styles.terminal} ref={logRef}>
                    {log.length === 0 ? (
                        <pre style={styles.pre}>{`╔══════════════════════════════════════════════════╗\n║          AKSHIT'S ENHANCED AI ASSISTANT       ║\n║           Powered by Advanced Integrations       ║\n╚══════════════════════════════════════════════════╝\n\nType a command and press Enter, e.g.:\n• summarize\n• show github repositories\n• analyze this job: https://linkedin.com/jobs/view/123\n`}</pre>
                    ) : (
                        log.map((line, i) => (
                            <pre key={`l-${i}`} style={styles.pre}>{line}</pre>
                        ))
                    )}
                </div>
                <div style={styles.inputRow}>
                    <input
                        style={styles.input}
                        placeholder="Type a command…"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    <button style={styles.button} onMouseDown={send}>Send</button>
                </div>
            </div>
        </Window>
    );
};

const styles: StyleSheetCSS = {
    container: {
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        padding: 12,
        boxSizing: 'border-box',
    },
    header: {
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        width: '100%',
    },
    status: {
        marginLeft: 12,
    },
    terminal: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'black',
        color: 'white',
        border: '2px solid #222',
        padding: 12,
        overflowY: 'auto',
    },
    pre: {
        fontFamily: 'Consolas, monospace',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        overflowWrap: 'anywhere',
        margin: 0,
    },
    inputRow: {
        marginTop: 12,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
    },
    input: {
        flex: 1,
        padding: 8,
        border: '1px solid #444',
        backgroundColor: '#111',
        color: '#eee',
    },
    button: {
        marginLeft: 8,
        padding: '8px 12px',
        border: '1px solid #444',
        backgroundColor: '#222',
        color: '#eee',
        cursor: 'pointer',
        fontFamily: 'MSSerif',
    },
};

export default ChatbotApp;