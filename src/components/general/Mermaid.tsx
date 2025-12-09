import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
});

interface MermaidProps {
    chart: string;
    className?: string;
    style?: React.CSSProperties;
}

// Preprocessor to fix common Mermaid syntax issues
const preprocessMermaidChart = (chart: string): string => {
    if (!chart) return chart;

    // Fix 1: Quote node labels containing parentheses, asterisks, or brackets
    // Matches patterns like: NodeName[Label with (parentheses)] or NodeName["Already quoted"]
    // and ensures they're properly quoted
    let processed = chart.replace(
        /(\w+)(\[[^\]]*\])/g,
        (match, nodeId, labelPart) => {
            // Check if the label is already quoted
            const labelContent = labelPart.slice(1, -1); // Remove brackets
            if (labelContent.startsWith('"') && labelContent.endsWith('"')) {
                return match; // Already quoted, leave as is
            }

            // Check if label contains special characters that need quoting
            if (/[\(\)\*\[\]\{\}\|]/.test(labelContent)) {
                return `${nodeId}["${labelContent}"]`;
            }

            return match; // No special characters, leave as is
        }
    );

    // Fix 2: Replace standalone asterisks in node names/labels
    // This handles cases like "A* Algorithm" which should be "A-Star Algorithm" or quoted
    processed = processed.replace(/\bA\*\b/g, '"A*"');
    processed = processed.replace(/\bA\* ([A-Za-z]+)\b/g, '"A* $1"');

    // Fix 3: Ensure subgraph titles with special chars are quoted
    processed = processed.replace(
        /(subgraph\s+)([^\n]+)/g,
        (match, keyword, title) => {
            if (/[\(\)\*\[\]\{\}\|]/.test(title) && !(title.startsWith('"') && title.endsWith('"'))) {
                return `${keyword}"${title}"`;
            }
            return match;
        }
    );

    return processed;
};

const Mermaid: React.FC<MermaidProps> = ({ chart, className, style }) => {
    const [svg, setSvg] = useState<string>('');
    const [error, setError] = useState<string | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const transformRef = useRef({ x: 0, y: 0, scale: 1 });
    const dragRef = useRef({ isDragging: false, startX: 0, startY: 0 });

    useEffect(() => {
        if (chart) {
            setError(null);
            const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

            // Preprocess the chart to fix common syntax issues
            const processedChart = preprocessMermaidChart(chart);

            mermaid.render(id, processedChart).then((result) => {
                setSvg(result.svg);
            }).catch((error) => {
                console.error('Mermaid rendering failed:', error);
                setError(error.message || 'Failed to render diagram');
                setSvg(`
                    <div style="
                        color: #d32f2f; 
                        background: #ffebee; 
                        padding: 15px; 
                        border: 1px solid #ef5350; 
                        border-radius: 4px; 
                        font-family: monospace;
                        white-space: pre-wrap;
                        overflow: auto;
                        max-height: 200px;
                    ">
                        <strong>Mermaid Rendering Error:</strong><br/>
                        ${error.message || 'Unknown error'}<br/><br/>
                        <strong>Debug Info:</strong><br/>
                        ${JSON.stringify({
                    message: error.message,
                    str: error.str,
                    hash: error.hash
                }, null, 2)}
                    </div>
                `);
            });
        }
    }, [chart]);

    const updateTransform = () => {
        if (containerRef.current) {
            const { x, y, scale } = transformRef.current;
            containerRef.current.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
        }
    };

    useEffect(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) return;

        const onWheel = (e: WheelEvent) => {
            e.preventDefault();
            e.stopPropagation();

            const scaleAdjustment = e.deltaY * -0.001;
            const newScale = Math.min(Math.max(0.5, transformRef.current.scale + scaleAdjustment), 3);
            transformRef.current.scale = newScale;
            updateTransform();
        };

        wrapper.addEventListener('wheel', onWheel, { passive: false });

        return () => {
            wrapper.removeEventListener('wheel', onWheel);
        };
    }, []);

    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        dragRef.current.isDragging = true;
        dragRef.current.startX = e.clientX - transformRef.current.x;
        dragRef.current.startY = e.clientY - transformRef.current.y;
        if (wrapperRef.current) {
            wrapperRef.current.style.cursor = 'grabbing';
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (dragRef.current.isDragging) {
            e.preventDefault();
            e.stopPropagation();
            transformRef.current.x = e.clientX - dragRef.current.startX;
            transformRef.current.y = e.clientY - dragRef.current.startY;
            updateTransform();
        }
    };

    const handleMouseUp = () => {
        dragRef.current.isDragging = false;
        if (wrapperRef.current) {
            wrapperRef.current.style.cursor = 'grab';
        }
    };

    return (
        <div
            ref={wrapperRef}
            className={`mermaid-wrapper ${className || ''}`}
            style={{
                overflow: 'hidden',
                border: '1px solid #ccc',
                height: '400px',
                position: 'relative',
                cursor: 'grab',
                backgroundColor: '#f9f9f9',
                marginBottom: '20px',
                ...style
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
        >
            <div
                ref={containerRef}
                dangerouslySetInnerHTML={{ __html: svg }}
                style={{
                    transform: `translate(0px, 0px) scale(1)`,
                    transformOrigin: 'center',
                    transition: 'transform 0.05s linear',
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}
            />
            <div style={{
                position: 'absolute',
                bottom: 10,
                right: 10,
                background: 'rgba(255,255,255,0.8)',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                pointerEvents: 'none',
                border: '1px solid #ddd'
            }}>
                Scroll to Zoom • Drag to Pan
            </div>
        </div>
    );
};

export default Mermaid;