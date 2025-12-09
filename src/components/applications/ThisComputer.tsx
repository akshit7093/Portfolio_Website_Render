import React from 'react';
import Window from '../os/Window';
import Icon from '../general/Icon';

export interface ThisComputerProps extends WindowAppProps { }

const ThisComputerApp: React.FC<ThisComputerProps> = (props) => {
    const openProjects = () => {
        if (props.onOpenApp) {
            props.onOpenApp('projects');
        }
    };

    const openChatbot = () => {
        if (props.onOpenApp) {
            props.onOpenApp('chatbot');
        }
    };

    const downloadResume = () => {
        // In a real app, this would trigger a download
        window.open('/resume.pdf', '_blank');
    };

    return (
        <Window
            top={20}
            left={20}
            width={600}
            height={400}
            windowBarIcon="computerSmall"
            windowTitle="This Computer"
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
        >
            <div style={styles.container}>
                <div style={styles.sidebar}>
                    <div style={styles.sidebarHeader}>
                        <div style={styles.sidebarTitle}>System Tasks</div>
                    </div>
                    <div style={styles.sidebarContent}>
                        <div style={styles.sidebarItem} onClick={() => alert("System: AkshitOS v1.0\nUser: Akshit Sharma\nRole: Full Stack Developer")}>View system information</div>
                        <div style={styles.sidebarItem} onClick={props.onClose}>Close window</div>
                    </div>
                    <div style={styles.sidebarHeader}>
                        <div style={styles.sidebarTitle}>Other Places</div>
                    </div>
                    <div style={styles.sidebarContent}>
                        <div style={styles.sidebarItem} onClick={openProjects}>My Network Places</div>
                        <div style={styles.sidebarItem} onClick={downloadResume}>My Documents</div>
                    </div>
                    <div style={styles.sidebarHeader}>
                        <div style={styles.sidebarTitle}>Details</div>
                    </div>
                    <div style={styles.sidebarContent}>
                        <div style={{ fontSize: 10, color: '#000' }}>
                            <b>This Computer</b><br />
                            System Folder
                        </div>
                    </div>
                </div>
                <div style={styles.mainContent}>
                    <div style={styles.header}>
                        <div style={styles.headerLabel}>Address</div>
                        <div style={styles.addressBar}>
                            <Icon icon="computerSmall" size={16} style={{ marginRight: 4 }} />
                            This Computer
                        </div>
                    </div>
                    <div style={styles.drivesContainer}>
                        <div style={styles.driveItem} onDoubleClick={openProjects}>
                            <Icon icon="myComputer" size={48} />
                            <div style={styles.driveLabel}>Projects (C:)</div>
                            <div style={styles.driveInfo}>Local Disk</div>
                        </div>
                        <div style={styles.driveItem} onDoubleClick={downloadResume}>
                            <Icon icon="cd" size={48} />
                            <div style={styles.driveLabel}>Resume (A:)</div>
                            <div style={styles.driveInfo}>3½ Floppy</div>
                        </div>
                        <div style={styles.driveItem} onDoubleClick={openChatbot}>
                            <Icon icon="chatbot" size={48} />
                            <div style={styles.driveLabel}>Contact (D:)</div>
                            <div style={styles.driveInfo}>Network Drive</div>
                        </div>
                    </div>
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
        backgroundColor: '#fff',
    },
    sidebar: {
        width: 200,
        background: 'linear-gradient(to bottom, #7b92e3 0%, #7b92e3 100%)',
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
    },
    sidebarHeader: {
        marginBottom: 4,
    },
    sidebarTitle: {
        fontWeight: 'bold',
        color: '#fff',
        fontSize: 14,
    },
    sidebarContent: {
        backgroundColor: '#fff',
        padding: 8,
        marginBottom: 16,
        border: '1px solid #fff',
        opacity: 0.9,
    },
    sidebarItem: {
        fontSize: 12,
        color: '#000080',
        cursor: 'pointer',
        marginBottom: 4,
        textDecoration: 'underline',
    },
    mainContent: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#fff',
    },
    header: {
        display: 'flex',
        alignItems: 'center',
        padding: 8,
        borderBottom: '1px solid #ccc',
    },
    headerLabel: {
        marginRight: 8,
        color: '#666',
        fontSize: 12,
    },
    addressBar: {
        flex: 1,
        border: '1px solid #ccc',
        padding: '2px 4px',
        display: 'flex',
        alignItems: 'center',
        backgroundColor: '#fff',
        fontSize: 12,
    },
    drivesContainer: {
        display: 'flex',
        flexDirection: 'row',
        flexWrap: 'wrap',
        padding: 20,
        gap: 32,
    },
    driveItem: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        width: 80,
        cursor: 'pointer',
    },
    driveLabel: {
        marginTop: 4,
        fontSize: 12,
        textAlign: 'center',
        color: '#000',
    },
    driveInfo: {
        fontSize: 10,
        color: '#666',
        textAlign: 'center',
    },
};

export default ThisComputerApp;
