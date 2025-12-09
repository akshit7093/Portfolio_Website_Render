import React, { useEffect, useRef, useState } from 'react';
import { IconName } from '../../assets/icons';
import colors from '../../constants/colors';
import Colors from '../../constants/colors';
import Icon from '../general/Icon';
import Button from './Button';
import DragIndicator from './DragIndicator';
import ResizeIndicator from './ResizeIndicator';
import DockPreview from './DockPreview';

export interface WindowProps {
    closeWindow: () => void;
    minimizeWindow: () => void;
    onInteract: () => void;
    width: number;
    height: number;
    top: number;
    left: number;
    windowTitle?: string;
    bottomLeftText?: string;
    rainbow?: boolean;
    windowBarColor?: string;
    windowBarIcon?: IconName;
    onWidthChange?: (width: number) => void;
    onHeightChange?: (height: number) => void;
    onToggleSidebar?: () => void;
}

const Window: React.FC<WindowProps> = (props) => {
    const windowRef = useRef<any>(null);
    const dragRef = useRef<any>(null);
    const contentRef = useRef<any>(null);

    const dragProps = useRef<{
        dragStartX: any;
        dragStartY: any;
    }>();

    const resizeRef = useRef<any>(null);

    // Detect mobile device
    const isMobile = window.innerWidth <= 768;

    // Initialize state with mobile check
    const [top, setTop] = useState(isMobile ? 0 : props.top);
    const [left, setLeft] = useState(isMobile ? 0 : props.left);
    const [width, setWidth] = useState(isMobile ? window.innerWidth : props.width);
    const [height, setHeight] = useState(isMobile ? window.innerHeight - 32 : props.height);

    const lastClickInside = useRef(false);

    const [contentWidth, setContentWidth] = useState(props.width);
    const [contentHeight, setContentHeight] = useState(props.height);

    const [windowActive, setWindowActive] = useState(true);

    const [isMaximized, setIsMaximized] = useState(false);
    const [preMaxSize, setPreMaxSize] = useState({
        width,
        height,
        top,
        left,
    });

    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);

    const DOCK_THRESHOLD = 32;
    const [snapTarget, setSnapTarget] = useState<'left' | 'right' | 'top' | null>(null);
    const [dockRect, setDockRect] = useState<{
        top: number;
        left: number;
        width: number;
        height: number;
    } | null>(null);

    // Determine which dock zone the cursor is in for given coordinates
    const getSnapTargetFromCoords = (
        clientX: number,
        clientY: number
    ): 'left' | 'right' | 'top' | null => {
        if (clientX <= DOCK_THRESHOLD) return 'left';
        if (clientX >= window.innerWidth - DOCK_THRESHOLD) return 'right';
        if (clientY <= DOCK_THRESHOLD) return 'top';
        return null;
    };

    const startResize = (event: any) => {
        event.preventDefault();
        setIsResizing(true);
        window.addEventListener('mousemove', onResize, false);
        window.addEventListener('mouseup', stopResize, false);
    };

    const requestRef = useRef<number>();

    const onResize = ({ clientX, clientY }: any) => {
        if (requestRef.current) return;

        requestRef.current = requestAnimationFrame(() => {
            const curWidth = clientX - left;
            const curHeight = clientY - top;
            if (curWidth > 520) resizeRef.current.style.width = `${curWidth}px`;
            if (curHeight > 220) resizeRef.current.style.height = `${curHeight}px`;
            resizeRef.current.style.opacity = 1;
            requestRef.current = undefined;
        });
    };

    const stopResize = () => {
        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }
        setIsResizing(false);
        setWidth(resizeRef.current.style.width);
        setHeight(resizeRef.current.style.height);
        resizeRef.current.style.opacity = 0;
        window.removeEventListener('mousemove', onResize, false);
        window.removeEventListener('mouseup', stopResize, false);
    };

    const startDrag = (event: any) => {
        const { clientX, clientY } = event;
        setIsDragging(true);
        event.preventDefault();
        dragProps.current = {
            dragStartX: clientX,
            dragStartY: clientY,
        };
        window.addEventListener('mousemove', onDrag, false);
        window.addEventListener('mouseup', stopDrag, false);
    };

    // Touch Event Handlers
    const startTouchDrag = (event: any) => {
        const touch = event.touches[0];
        const { clientX, clientY } = touch;
        setIsDragging(true);
        dragProps.current = {
            dragStartX: clientX,
            dragStartY: clientY,
        };
        window.addEventListener('touchmove', onTouchDrag, { passive: false });
        window.addEventListener('touchend', stopTouchDrag, false);
    };

    const onTouchDrag = (event: any) => {
        if (event.cancelable) {
            event.preventDefault(); // Prevent scrolling while dragging
        }
        const touch = event.touches[0];
        onDrag({ clientX: touch.clientX, clientY: touch.clientY });
    };

    const stopTouchDrag = (event: any) => {
        const touch = event.changedTouches[0];
        stopDrag({ clientX: touch.clientX, clientY: touch.clientY });
        window.removeEventListener('touchmove', onTouchDrag);
        window.removeEventListener('touchend', stopTouchDrag);
    };

    const getDockRect = (
        target: 'left' | 'right' | 'top'
    ): { top: number; left: number; width: number; height: number } => {
        const fullWidth = window.innerWidth;
        const fullHeight = window.innerHeight - 32;
        if (target === 'left') {
            return { top: 0, left: 0, width: Math.floor(fullWidth / 2), height: fullHeight };
        }
        if (target === 'right') {
            return { top: 0, left: Math.floor(fullWidth / 2), width: Math.floor(fullWidth / 2), height: fullHeight };
        }
        // top -> maximize
        return { top: 0, left: 0, width: fullWidth, height: fullHeight };
    };

    const onDrag = ({ clientX, clientY }: any) => {
        if (requestRef.current) return;

        requestRef.current = requestAnimationFrame(() => {
            let { x, y } = getXYFromDragProps(clientX, clientY);
            // Use translate3d for GPU acceleration
            dragRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
            dragRef.current.style.opacity = 1;

            const target = getSnapTargetFromCoords(clientX, clientY);
            setSnapTarget(target);
            setDockRect(target ? getDockRect(target) : null);
            requestRef.current = undefined;
        });
    };

    const stopDrag = ({ clientX, clientY }: any) => {
        if (requestRef.current) {
            cancelAnimationFrame(requestRef.current);
            requestRef.current = undefined;
        }
        setIsDragging(false);
        const { x, y } = getXYFromDragProps(clientX, clientY);
        // Recompute target on mouseup to avoid any stale state race
        const target = getSnapTargetFromCoords(clientX, clientY);
        if (target) {
            const rect = getDockRect(target);
            if (target === 'top') {
                setPreMaxSize({ width, height, top, left });
            }
            setWidth(rect.width);
            setHeight(rect.height);
            setTop(rect.top);
            setLeft(rect.left);
            setIsMaximized(target === 'top');
        } else {
            setTop(y);
            setLeft(x);
        }

        setSnapTarget(null);
        setDockRect(null);

        window.removeEventListener('mousemove', onDrag, false);
        window.removeEventListener('mouseup', stopDrag, false);
    };

    const getXYFromDragProps = (
        clientX: number,
        clientY: number
    ): { x: number; y: number } => {
        if (!dragProps.current) return { x: 0, y: 0 };
        const { dragStartX, dragStartY } = dragProps.current;

        const x = clientX - dragStartX + left;
        const y = clientY - dragStartY + top;

        return { x, y };
    };

    useEffect(() => {
        // Use translate3d for GPU acceleration
        dragRef.current.style.transform = `translate3d(${left}px, ${top}px, 0)`;
    });

    useEffect(() => {
        props.onWidthChange && props.onWidthChange(contentWidth);
    }, [props.onWidthChange, contentWidth]); // eslint-disable-line

    useEffect(() => {
        props.onHeightChange && props.onHeightChange(contentHeight);
    }, [props.onHeightChange, contentHeight]); // eslint-disable-line

    useEffect(() => {
        setContentWidth(contentRef.current.getBoundingClientRect().width);
    }, [width]);

    useEffect(() => {
        setContentHeight(contentRef.current.getBoundingClientRect().height);
    }, [height]);

    const maximize = () => {
        if (isMaximized) {
            setWidth(preMaxSize.width);
            setHeight(preMaxSize.height);
            setTop(preMaxSize.top);
            setLeft(preMaxSize.left);
            setIsMaximized(false);
        } else {
            setPreMaxSize({
                width,
                height,
                top,
                left,
            });
            setWidth(window.innerWidth);
            setHeight(window.innerHeight - 32);
            setTop(0);
            setLeft(0);
            setIsMaximized(true);
        }
    };

    const onCheckClick = () => {
        if (lastClickInside.current) {
            setWindowActive(true);
        } else {
            setWindowActive(false);
        }
        lastClickInside.current = false;
    };

    useEffect(() => {
        window.addEventListener('mousedown', onCheckClick, false);
        return () => {
            window.removeEventListener('mousedown', onCheckClick, false);
        };
    }, []);

    const onWindowInteract = () => {
        props.onInteract();
        setWindowActive(true);
        lastClickInside.current = true;
    };

    const mobileStyles = isMobile ? {
        width: '100vw',
        height: 'calc(100vh - 32px)',
        // We don't force top/left/position here anymore to allow dragging
        // But we initialize them correctly in useState
    } : {
        width,
        height,
        top,
        left,
    };

    return (
        <div onMouseDown={onWindowInteract} onTouchStart={onWindowInteract} style={styles.container}>
            <div
                className="os-window"
                style={Object.assign({}, styles.window, {
                    width,
                    height,
                    top,
                    left,
                    willChange: isDragging || isResizing ? 'transform, width, height' : 'auto',
                })}
                ref={windowRef}
            >
                <div style={styles.windowBorderOuter}>
                    <div style={styles.windowBorderInner}>
                        <div
                            style={styles.dragHitbox}
                            onMouseDown={startDrag}
                            onTouchStart={startTouchDrag}
                        ></div>
                        <div
                            className={props.rainbow ? 'rainbow-wrapper' : ''}
                            style={Object.assign(
                                {},
                                styles.topBar,
                                props.windowBarColor && {
                                    backgroundColor: props.windowBarColor,
                                },
                                !windowActive && {
                                    backgroundColor: Colors.darkGray,
                                }
                            )}
                        >
                            <div style={styles.windowHeader}>
                                {props.onToggleSidebar && (
                                    <button
                                        onClick={props.onToggleSidebar}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        onTouchStart={(e) => e.stopPropagation()}
                                        onDoubleClick={(e) => e.stopPropagation()}
                                        style={{
                                            background: 'transparent',
                                            border: 'none',
                                            color: Colors.white,
                                            fontSize: 18,
                                            fontWeight: 'bold',
                                            cursor: 'pointer',
                                            padding: '0 8px',
                                            marginRight: 4,
                                            display: 'flex',
                                            alignItems: 'center',
                                            height: '100%',
                                            position: 'relative',
                                            zIndex: 10002,
                                        }}
                                    >
                                        ☰
                                    </button>
                                )}
                                {props.windowBarIcon ? (
                                    <Icon
                                        icon={props.windowBarIcon}
                                        style={Object.assign(
                                            {},
                                            styles.windowBarIcon,
                                            !windowActive && { opacity: 0.5 }
                                        )}
                                        size={16}
                                    />
                                ) : (
                                    <div style={{ width: 16 }} />
                                )}
                                <p
                                    style={
                                        windowActive
                                            ? { whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }
                                            : { color: colors.lightGray, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }
                                    }
                                    className="showcase-header"
                                >
                                    {props.windowTitle}
                                </p>
                            </div>
                            <div style={styles.windowTopButtons}>
                                <Button
                                    icon="minimize"
                                    onClick={props.minimizeWindow}
                                />
                                <Button icon="maximize" onClick={maximize} />
                                <div style={{ paddingLeft: 2 }}>
                                    <Button
                                        icon="close"
                                        onClick={props.closeWindow}
                                    />
                                </div>
                            </div>
                        </div>
                        <div
                            style={Object.assign({}, styles.contentOuter, {
                                // zIndex: isDragging || isResizing ? 0 : 100,
                            })}
                        >
                            <div style={styles.contentInner}>
                                <div style={styles.content} ref={contentRef}>
                                    {props.children}
                                </div>
                            </div>
                        </div>
                        <div
                            onMouseDown={startResize}
                            style={styles.resizeHitbox}
                        ></div>
                        <div style={styles.bottomBar}>
                            <div
                                style={Object.assign({}, styles.insetBorder, {
                                    flex: 5 / 7,
                                    alignItems: 'center',
                                })}
                            >
                                <p
                                    style={{
                                        fontSize: 12,
                                        marginLeft: 4,
                                        fontFamily: 'MSSerif',
                                    }}
                                >
                                    {props.bottomLeftText}
                                </p>
                            </div>
                            <div
                                style={Object.assign(
                                    {},
                                    styles.insetBorder,
                                    styles.bottomSpacer
                                )}
                            />
                            <div
                                style={Object.assign(
                                    {},
                                    styles.insetBorder,
                                    styles.bottomSpacer
                                )}
                            />
                            <div
                                style={Object.assign(
                                    {},
                                    styles.insetBorder,
                                    styles.bottomResizeContainer
                                )}
                            >
                                <div
                                    style={{
                                        alignItems: 'flex-end',
                                    }}
                                >
                                    <Icon size={12} icon="windowResize" />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div
                style={
                    !isResizing
                        ? {
                            zIndex: -10000,
                            pointerEvents: 'none',
                        }
                        : {
                            zIndex: 1000,
                            cursor: 'nwse-resize',
                            mixBlendMode: 'difference',
                        }
                }
            >
                <ResizeIndicator
                    top={top}
                    left={left}
                    width={width}
                    height={height}
                    resizeRef={resizeRef}
                />
            </div>
            <div
                style={
                    !isDragging
                        ? {
                            zIndex: -10000,
                            pointerEvents: 'none',
                        }
                        : {
                            zIndex: 1000,
                            cursor: 'move',
                            mixBlendMode: 'difference',
                        }
                }
            >
                <DragIndicator
                    width={width}
                    height={height}
                    dragRef={dragRef}
                />
            </div>
            {isDragging && (
                <DockPreview rect={dockRect} />
            )}
        </div>
    );
};

const styles: StyleSheetCSS = {
    window: {
        backgroundColor: Colors.lightGray,
        position: 'absolute',
        boxShadow: '4px 4px 0px rgba(0, 0, 0, 0.2)', // Add shadow for better visibility
        display: 'flex',
        flexDirection: 'column',
        boxSizing: 'border-box',
    },
    container: {
        position: 'absolute', // Make container absolute to match window behavior
        top: 0,
        left: 0,
        width: 0,
        height: 0,
        overflow: 'visible',
    },
    dragHitbox: {
        position: 'absolute',
        width: 'calc(100% - 70px)',
        height: 48,
        zIndex: 10000,
        top: -8,
        left: -4,
        cursor: 'move',
        touchAction: 'none', // Prevent browser handling of gestures
    },
    windowBorderOuter: {
        border: `1px solid ${Colors.black}`,
        borderTopColor: colors.lightGray,
        borderLeftColor: colors.lightGray,
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0, // Ensure it can shrink
        boxSizing: 'border-box',
    },
    windowBorderInner: {
        border: `1px solid ${Colors.darkGray}`,
        borderTopColor: colors.white,
        borderLeftColor: colors.white,
        flex: 1,
        padding: 2,

        flexDirection: 'column',
        display: 'flex',
        minHeight: 0, // Ensure it can shrink
        boxSizing: 'border-box',
    },
    resizeHitbox: {
        position: 'absolute',
        width: 60,
        height: 60,
        bottom: -20,
        right: -20,
        cursor: 'nwse-resize',
        touchAction: 'none',
    },
    topBar: {
        display: 'flex',
        backgroundColor: Colors.blue,
        width: '100%',
        height: 20,
        flexShrink: 0, // Prevent top bar from shrinking

        alignItems: 'center',
        paddingRight: 2,
        boxSizing: 'border-box',
    },
    contentOuter: {
        border: `1px solid ${Colors.white}`,
        borderTopColor: colors.darkGray,
        borderLeftColor: colors.darkGray,
        flexGrow: 1,
        flexShrink: 1, // Allow content to shrink if needed
        flexBasis: 0, // Force flex to calculate size based on constraint

        marginTop: 8,
        marginBottom: 8,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        boxSizing: 'border-box',
    },
    contentInner: {
        border: `1px solid ${Colors.lightGray}`,
        borderTopColor: colors.black,
        borderLeftColor: colors.black,
        flex: 1,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        boxSizing: 'border-box',
        flexBasis: 0,
    },
    content: {
        flex: 1,

        position: 'relative',
        overflowY: 'auto', // Enable vertical scrolling
        overflowX: 'hidden',
        backgroundColor: Colors.white,
        minWidth: 0, // Prevent flex item from overflowing
        minHeight: 0,
        boxSizing: 'border-box',
        flexBasis: 0,
    },
    bottomBar: {
        flexShrink: 0, // Prevent bottom bar from shrinking
        width: '100%',
        height: 20,
    },
    bottomSpacer: {
        width: 16,
        marginLeft: 2,
    },
    insetBorder: {
        border: `1px solid ${Colors.white}`,
        borderTopColor: colors.darkGray,
        borderLeftColor: colors.darkGray,
        padding: 2,
    },
    bottomResizeContainer: {
        flex: 2 / 7,

        justifyContent: 'flex-end',
        padding: 0,
        marginLeft: 2,
    },
    windowTopButtons: {
        // zIndex: 10000,
        display: 'flex',
        alignItems: 'center',
        flexShrink: 0,
        minWidth: 60,
        justifyContent: 'flex-end',
    },
    windowHeader: {
        flex: 1,
        // justifyContent: 'center',
        // alignItems: 'center',
    },
    windowBarIcon: {
        paddingLeft: 4,
        paddingRight: 4,
    },
};

export default Window;
