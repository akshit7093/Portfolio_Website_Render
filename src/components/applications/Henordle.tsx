import React from 'react';
import Window from '../os/Window';
import Wordle from '../wordle/Wordle';

export interface HenordleAppProps extends WindowAppProps { }

const HenordleApp: React.FC<HenordleAppProps> = (props) => {
    return (
        <Window
            top={20}
            left={window.innerWidth <= 768 ? 0 : 300}
            width={window.innerWidth <= 768 ? window.innerWidth : 600}
            height={window.innerWidth <= 768 ? window.innerHeight : 860}
            windowBarIcon="windowGameIcon"
            windowTitle="Henordle"
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
            bottomLeftText={'© Copyright 2022 Akshit Sharma'}
        >
            <div className="site-page">
                <Wordle />
            </div>
        </Window>
    );
};

export default HenordleApp;
