import React from 'react';
import Colors from '../../constants/colors';

export interface DockPreviewProps {
    rect: { top: number; left: number; width: number; height: number } | null;
}

const DockPreview: React.FC<DockPreviewProps> = ({ rect }) => {
    if (!rect) return null;
    const { top, left, width, height } = rect;

    return (
        <div
            style={Object.assign({}, styles.overlay)}
        >
            <div
                style={Object.assign({}, styles.previewRect, {
                    top,
                    left,
                    width,
                    height,
                })}
            />
        </div>
    );
};

const BORDER_WIDTH = 3;

export const styles: StyleSheetCSS = {
    overlay: {
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: 1000,
        mixBlendMode: 'difference',
    },
    previewRect: {
        position: 'absolute',
        border: `${BORDER_WIDTH}px solid ${Colors.white}`,
        boxSizing: 'border-box',
        backgroundColor: 'rgba(255,255,255,0.06)',
    },
};

export default DockPreview;