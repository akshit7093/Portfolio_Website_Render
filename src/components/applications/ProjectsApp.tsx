import React, { useState, useEffect } from 'react';
import Window from '../os/Window';
import OpenStack from '../showcase/projects/software/OpenStack';
import PriceIntelligence from '../showcase/projects/software/PriceIntelligence';
import Chatbot from '../showcase/projects/software/Chatbot';
import useInitialWindowSize from '../../hooks/useInitialWindowSize';

export interface ProjectsAppProps extends WindowAppProps { }

const ProjectsApp: React.FC<ProjectsAppProps> = (props) => {
    const { initWidth, initHeight } = useInitialWindowSize({ margin: 100 });
    const [currentView, setCurrentView] = useState<string>('openstack');
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

    const renderContent = () => {
        switch (currentView) {
            case 'openstack':
                return <OpenStack />;
            case 'price-intelligence':
                return <PriceIntelligence />;
            case 'chatbot':
                return <Chatbot />;
            default:
                return <OpenStack />;
        }
    };

    const navItems = [
        { id: 'openstack', label: 'OpenStack Cloud Manager' },
        { id: 'price-intelligence', label: 'Price Intelligence System' },
        { id: 'chatbot', label: 'Universal Chatbot' },
    ];

    const isMobile = window.innerWidth <= 768;

    return (
        <Window
            top={24}
            left={56}
            width={initWidth}
            height={initHeight}
            windowTitle="Projects Explorer"
            windowBarIcon="computerSmall"
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
            bottomLeftText={'© Copyright 2025 Akshit Sharma'}
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
                    <h3 style={styles.sidebarHeader}>Projects</h3>
                    <div style={styles.navLinks}>
                        {navItems.map((item) => (
                            <div
                                key={item.id}
                                style={{
                                    ...styles.navItem,
                                    ...(currentView === item.id ? styles.activeNavItem : {}),
                                }}
                                onClick={() => {
                                    setCurrentView(item.id);
                                    if (isMobile) setSidebarOpen(false);
                                }}
                            >
                                {item.label}
                            </div>
                        ))}
                    </div>
                </div>
                <div style={styles.content}>
                    {renderContent()}
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
        padding: '0',
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

export default ProjectsApp;
