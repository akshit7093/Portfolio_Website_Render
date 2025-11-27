import React, { useState, useEffect } from 'react';
import Window from '../os/Window';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import Mermaid from '../general/Mermaid';
import useInitialWindowSize from '../../hooks/useInitialWindowSize';
import 'katex/dist/katex.min.css';

export interface ResearchPapersAppProps extends WindowAppProps { }

const ResearchPapersApp: React.FC<ResearchPapersAppProps> = (props) => {
    const { initWidth, initHeight } = useInitialWindowSize({ margin: 100 });
    const [papers, setPapers] = useState<{ id: string; title: string; content: string }[]>([]);
    const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);

    useEffect(() => {
        // Dynamically load all .md files from src/assets/research-papers
        // require.context is a webpack feature
        const importAll = (r: any) => {
            return r.keys().map((fileName: string) => {
                const contentPath = r(fileName) as string;

                return {
                    id: fileName.replace('./', '').replace('.md', ''),
                    title: fileName.replace('./', '').replace('.md', '').replace(/-/g, ' '),
                    path: contentPath
                };
            });
        };

        const loadPapers = async () => {
            // @ts-ignore
            const context = require.context('../../assets/research-papers', false, /\.md$/);
            const paperMetadata = importAll(context);

            const loadedPapers = await Promise.all(
                paperMetadata.map(async (meta: { id: string; title: string; path: string }) => {
                    const response = await fetch(meta.path);
                    const text = await response.text();

                    // Preprocess LaTeX: Fix corrupted backslashes in formulas
                    // Handles cases where \t, \f, \r, \b were interpreted as control characters
                    const fixedText = text
                        .replace(/[\t\s]ext\{/g, '\\text{')      // Fix \text (tab/space + ext)
                        .replace(/[\x0c]rac\{/g, '\\frac{')      // Fix \frac (form feed + rac)
                        .replace(/[\t]imes/g, '\\times')         // Fix \times (tab + imes)
                        .replace(/[\r\n]?ight\)/g, '\\right)')   // Fix \right) (CR/LF + ight)
                        .replace(/eft\(/g, '\\left(')            // Fix \left( (eft)
                        .replace(/([^\\])hat\{/g, '$1\\hat{')    // Fix \hat (if backslash missing)
                        .replace(/([^\\])sum_/g, '$1\\sum_')     // Fix \sum (if backslash missing)
                        .replace(/–/g, '-');                     // Fix en-dashes in math

                    return {
                        id: meta.id,
                        title: meta.title,
                        content: fixedText
                    };
                })
            );

            setPapers(loadedPapers);
            if (loadedPapers.length > 0) {
                setSelectedPaperId(loadedPapers[0].id);
            }
        };

        loadPapers();
    }, []);

    const selectedPaper = papers.find(p => p.id === selectedPaperId);
    const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);

    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth > 768) {
                setSidebarOpen(true);
            } else {
                setSidebarOpen(false);
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    const isMobile = window.innerWidth <= 768;

    return (
        <Window
            top={24}
            left={56}
            width={initWidth}
            height={initHeight}
            windowTitle="Research Papers"
            windowBarIcon="showcaseIcon" // Using generic icon for now
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
            bottomLeftText={'© Research Archive'}
        >
            <div style={styles.container}>
                {/* Hamburger Menu Button (Mobile Only) */}
                {isMobile && (
                    <button
                        onClick={toggleSidebar}
                        style={styles.hamburger}
                        aria-label="Toggle navigation"
                    >
                        ☰
                    </button>
                )}

                <div style={Object.assign({}, styles.sidebar,
                    isMobile && !sidebarOpen && { transform: 'translateX(-100%)', position: 'absolute', height: '100%', zIndex: 1000 },
                    isMobile && sidebarOpen && { transform: 'translateX(0)', position: 'absolute', height: '100%', zIndex: 1000, boxShadow: '2px 0 5px rgba(0,0,0,0.5)' }
                )}>
                    {isMobile && (
                        <button onClick={() => setSidebarOpen(false)} style={styles.closeButton}>×</button>
                    )}
                    <h3 style={styles.sidebarHeader}>Papers</h3>
                    <div style={styles.navLinks}>
                        {papers.map((paper) => (
                            <div
                                key={paper.id}
                                style={{
                                    ...styles.navItem,
                                    ...(selectedPaperId === paper.id ? styles.activeNavItem : {}),
                                }}
                                onClick={() => {
                                    setSelectedPaperId(paper.id);
                                    if (isMobile) setSidebarOpen(false);
                                }}
                            >
                                {paper.title}
                            </div>
                        ))}
                    </div>
                </div>
                <div style={styles.content}>
                    {selectedPaper ? (
                        <div className="markdown-content" style={{
                            padding: isMobile ? '16px' : '32px',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '16px',
                            overflowWrap: 'break-word',
                            wordWrap: 'break-word',
                            maxWidth: '100%'
                        }}>
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm, remarkMath] as any}
                                rehypePlugins={[rehypeKatex] as any}
                                components={{
                                    code({ node, inline, className, children, ...props }: any) {
                                        const match = /language-(\w+)/.exec(className || '');
                                        const codeContent = String(children).replace(/\n$/, '');

                                        if (!inline && match && match[1] === 'mermaid') {
                                            return <Mermaid chart={codeContent} />;
                                        }

                                        return !inline && match ? (
                                            <pre className={className} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                                <code {...props} className={className}>
                                                    {children}
                                                </code>
                                            </pre>
                                        ) : (
                                            <code className={className} {...props}>
                                                {children}
                                            </code>
                                        );
                                    },
                                    // Ensure images are responsive
                                    img: ({ node, ...props }) => <img style={{ maxWidth: '100%', height: 'auto' }} {...props} />
                                }}
                            >
                                {selectedPaper.content}
                            </ReactMarkdown>
                        </div>
                    ) : (
                        <div style={{ padding: '32px' }}>Loading papers...</div>
                    )}
                </div>
            </div>
        </Window>
    );
};

const styles: StyleSheetCSS = {
    container: {
        display: 'flex',
        flexDirection: 'row',
        height: '100%',
        width: '100%',
        backgroundColor: '#fff',
        position: 'relative',
        overflow: 'hidden',
    },
    sidebar: {
        width: '250px',
        borderRight: '2px solid #000',
        padding: '32px 16px',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#c0c0c0',
        flexShrink: 0,
        transition: 'transform 0.3s ease',
    },
    sidebarHeader: {
        marginBottom: '24px',
        fontFamily: 'MSSerif',
        fontSize: '24px',
        borderBottom: '2px solid #000',
        paddingBottom: '8px',
        marginTop: '20px',
    },
    navLinks: {
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
    },
    navItem: {
        cursor: 'pointer',
        fontFamily: 'MSSerif',
        fontSize: '16px',
        color: '#000080',
        textDecoration: 'underline',
    },
    activeNavItem: {
        fontWeight: 'bold',
        color: '#0000FF',
    },
    content: {
        flex: 1,
        overflowY: 'auto',
        padding: '0', // Padding handled by inner div
        backgroundColor: '#fff',
    },
    hamburger: {
        position: 'absolute',
        top: 8,
        left: 8,
        zIndex: 1001,
        fontSize: 24,
        background: '#f0f0f0',
        border: '1px solid #ccc',
        borderRadius: 4,
        padding: '4px 8px',
        cursor: 'pointer',
    },
    closeButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        fontSize: 24,
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
    }
};

export default ResearchPapersApp;
