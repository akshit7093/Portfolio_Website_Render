import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
});

interface MermaidProps {
    chart: string;
}

const Mermaid: React.FC<MermaidProps> = ({ chart }) => {
    const [svg, setSvg] = useState<string>('');
    const containerRef = useRef<HTMLDivElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const transformRef = useRef({ x: 0, y: 0, scale: 1 });
    const dragRef = useRef({ isDragging: false, startX: 0, startY: 0 });

    useEffect(() => {
        if (chart) {
            const id = `mermaid-${Date.now()}`;
            mermaid.render(id, chart).then((result) => {
                setSvg(result.svg);
            }).catch((error) => {
                console.error('Mermaid rendering failed:', error);
                setSvg('<div style="color: red; padding: 10px;">Failed to render diagram</div>');
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
            // Always prevent default to stop page scroll when over canvas
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
    };

    return (
        <div
            ref={wrapperRef}
            className="mermaid-wrapper"
            style={{
                overflow: 'hidden',
                border: '1px solid #ccc',
                height: '400px',
                position: 'relative',
                cursor: 'grab',
                backgroundColor: '#f9f9f9',
                marginBottom: '20px'
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
