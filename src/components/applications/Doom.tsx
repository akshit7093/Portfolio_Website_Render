import React, { useState } from 'react';
import DosPlayer from '../dos/DosPlayer';
import Window from '../os/Window';

export interface DoomAppProps extends WindowAppProps { }

const DoomApp: React.FC<DoomAppProps> = (props) => {
    const [width, setWidth] = useState(980);
    const [height, setHeight] = useState(670);
    const isMobile = window.innerWidth <= 768;

    return (
        <Window
            top={10}
            left={10}
            width={width}
            height={height}
            windowTitle="Doom"
            windowBarColor="#1C1C1C"
            windowBarIcon="windowGameIcon"
            bottomLeftText={'Powered by JSDOS & DOSBox'}
            closeWindow={props.onClose}
            onInteract={props.onInteract}
            minimizeWindow={props.onMinimize}
            onWidthChange={setWidth}
            onHeightChange={setHeight}
        >
            <DosPlayer width={isMobile ? '100%' : width} height={isMobile ? '100%' : height} bundleUrl="doom.jsdos" />
        </Window>
    );
};

export default DoomApp;
