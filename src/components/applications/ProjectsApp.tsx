import React, { useState } from 'react';
import Window from '../os/Window';
import OpenStack from '../showcase/projects/software/OpenStack';
import PriceIntelligence from '../showcase/projects/software/PriceIntelligence';
import Chatbot from '../showcase/projects/software/Chatbot';
import useInitialWindowSize from '../../hooks/useInitialWindowSize';

export interface ProjectsAppProps extends WindowAppProps { }

const ProjectsApp: React.FC<ProjectsAppProps> = (props) => {
    const { initWidth, initHeight } = useInitialWindowSize({ margin: 100 });
    const [currentView, setCurrentView] = useState<string>('openstack');

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
                <div style={styles.sidebar}>
                    <h3 style={styles.sidebarHeader}>Projects</h3>
                    <div style={styles.navLinks}>
                        {navItems.map((item) => (
                            <div
                                key={item.id}
                                style={{
                                    ...styles.navItem,
                                    ...(currentView === item.id ? styles.activeNavItem : {}),
                                }}
                                onClick={() => setCurrentView(item.id)}
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
    },
    sidebar: {
        width: '250px',
        borderRight: '2px solid #000',
        padding: '32px 16px',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#c0c0c0',
        flexShrink: 0,
    },
    sidebarHeader: {
        marginBottom: '24px',
        fontFamily: 'MSSerif',
        fontSize: '24px',
        borderBottom: '2px solid #000',
        paddingBottom: '8px',
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
};

export default ProjectsApp;
