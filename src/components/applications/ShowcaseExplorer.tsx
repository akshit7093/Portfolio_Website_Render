import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from '../showcase/Home';
import About from '../showcase/About';
import Window from '../os/Window';
import Experience from '../showcase/Experience';
import Projects from '../showcase/Projects';
import Contact from '../showcase/Contact';
import SoftwareProjects from '../showcase/projects/Software';
import Certifications from '../showcase/projects/Music';
import Achievements from '../showcase/projects/Art';
import VerticalNavbar from '../showcase/VerticalNavbar';
import useInitialWindowSize from '../../hooks/useInitialWindowSize';

export interface ShowcaseExplorerProps extends WindowAppProps { }

const ShowcaseExplorer: React.FC<ShowcaseExplorerProps> = (props) => {
    const { initWidth, initHeight } = useInitialWindowSize({ margin: 100 });
    const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);

    // Listen for window resize to auto-show sidebar on desktop
    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth > 768) {
                setSidebarOpen(true);
            }
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

    return (
        <Window
            top={24}
            left={56}
            width={initWidth}
            height={initHeight}
            windowTitle="Akshit Sharma - Portfolio 2025"
            windowBarIcon="windowExplorerIcon"
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
            bottomLeftText={'© Copyright 2025 Akshit Sharma'}
        >
            <Router>
                <div className="site-page">
                    {/* Hamburger Menu Button (Mobile Only) */}
                    {window.innerWidth <= 768 && (
                        <button
                            onClick={toggleSidebar}
                            style={styles.hamburger}
                            aria-label="Toggle navigation"
                        >
                            ☰
                        </button>
                    )}

                    <VerticalNavbar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/about" element={<About />} />
                        <Route path="/experience" element={<Experience />} />
                        <Route path="/projects" element={<Projects />} />
                        <Route path="/contact" element={<Contact />} />
                        <Route
                            path="/projects/software"
                            element={<SoftwareProjects />}
                        />
                        <Route
                            path="/projects/music"
                            element={<Certifications />}
                        />
                        <Route path="/projects/art" element={<Achievements />} />
                    </Routes>
                </div>
            </Router>
        </Window>
    );
};

const styles: StyleSheetCSS = {
    hamburger: {
        position: 'fixed',
        top: 8,
        left: 8,
        zIndex: 10000,
        fontSize: 24,
        background: '#f0f0f0',
        border: '1px solid #ccc',
        borderRadius: 4,
        padding: '8px 12px',
        cursor: 'pointer',
        fontFamily: 'Arial, sans-serif',
    },
};

export default ShowcaseExplorer;
