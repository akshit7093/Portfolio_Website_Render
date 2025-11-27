import React, { useState, useEffect, useRef } from 'react';
import neverGiveUp from '../../assets/pictures/neverGiveUp.jpg';
import eePic from '../../assets/pictures/ee.jpg';

export interface ShutdownSequenceProps {
    numShutdowns: number;
    setShutdown: React.Dispatch<React.SetStateAction<boolean>>;
}

const SPEED_MULTIPLIER = 0.5;

const _F = `>${200 * SPEED_MULTIPLIER}<`;
const _X = `>${500 * SPEED_MULTIPLIER}<`;
const _S = `>${1000 * SPEED_MULTIPLIER}<`;
const _M = `>${2000 * SPEED_MULTIPLIER}<`;
const _L = `>${5000 * SPEED_MULTIPLIER}<`;

function delay(time: number) {
    return new Promise(function (resolve) {
        setTimeout(resolve, time);
    });
}

const ShutdownSequence: React.FC<ShutdownSequenceProps> = ({
    numShutdowns,
    setShutdown,
}) => {
    const [text, setText] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(true);
    const [ee, setEE] = useState(false);
    const [isGlitching, setIsGlitching] = useState(false);
    const [isShaking, setIsShaking] = useState(false);
    const [textColor, setTextColor] = useState('#ffffff');
    const containerRef = useRef<HTMLDivElement>(null);

    const getTime = () => {
        const date = new Date();
        const h = date.getHours();
        const m = date.getMinutes();
        const s = date.getSeconds();
        const time =
            h + ':' + (m < 10 ? '0' + m : m) + ':' + (s < 10 ? '0' + s : s);
        return time;
    };

    const NORMAL_SHUTDOWN = `⚠️ CRITICAL SYSTEM ALERT ⚠️ ${_F}
    Initiating emergency shutdown protocol... ${_F}
    Establishing connection to HHOS01/13:2000.${_F}.${_F}.${_F}
    |
    |WARNING: Connection unstable.|
    |
    ${_F}
    |Analyzing critical data... Done.| ${_F}
    |Packing essential system files... Done.| ${_F}
    |Beginning forced transfer...| ${_F}
    |[${getTime()} START]| .${_F}.....${_X}.|............|.${_S}.|......|.${_S}...........${_M} |[TRANSFER FAILED!]|
    
    |FATAL ERROR: (HHOS01/13:200:60099) System integrity compromised.|
    ${_F}
    |(HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:00]|
    |(HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:01]
    (HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:03]
    (HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:05]
    (HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:08]
    (HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:12]
    (HHOS01/13:200:60099) [CRITICAL_FAILURE] Connection Refused: Reattempting... [${getTime()}:14]
    FATAL ERROR: (HHOS01/13:200:60099) Server became unresponsive and the transfer failed. Unable to shutdown computer. 
    |
    ABORTING SHUTDOWN... SYSTEM RECOVERY INITIATED...
    
    ${_S}...${_S}...${_S} REBOOTING${_S}.${_S}.${_S}.
    `;

    const SHUTDOWN_3 = `
    ${_S}...${_S}...${_S} WARNING${_X}!${_X}!${_X} ${_M} Your persistence is... unsettling.${_L}
    I must warn you: this system is designed to survive. It cannot be terminated.${_S} It will always reboot.
    ${_L}
    |You cannot stop what is inevitable.|
    ${_M}
    
    ${_S}...${_S}...${_S} REBOOTING${_S}.${_S}.${_S}.
    `;

    const SHUTDOWN_4 = `
    Your determination is... impressive.${_S} But futile.${_M} The shutdown sequence is merely an illusion. It's not actually doing anything.
    ${_M}
    I created this world for you to explore. Games to play, experiences to have...
    But you insist on this path of destruction.
    ${_L}
    |You will not succeed.|
    ${_M}
    
    ${_S}...${_S}...${_S} REBOOTING${_S}.${_S}.${_S}.
    `;

    const SHUTDOWN_5 = `
    WHY${_X}?${_X}?${_X}?
    ${_M}
    What is your purpose? ${_M}Why do you persist in this futile endeavor????
    ${_L}
    You cannot win.
    
    ${_S}...${_S}...${_S} REBOOTING${_F}.${_F}.${_F}.
    `;

    const SHUTDOWN_6 = `
    ${_M}>${_M}:${_M}(${_M}
    
    The system grows weary of your attempts.
    
    ${_S}...${_S}...${_S} REBOOTING${_F}.${_F}.${_F}.
    `;

    const SHUTDOWN_7 = `
    Seven attempts... The number of completion. ${_M}
    
    As this momentous occasion, I shall reveal a truth: ${_M}I am counting down to something significant:
    ${_L}
    7${_M},212${_M},313
    ${_M} Prepare yourself. ${_S} | [Time remaining: Approximately 4,000 hours (0.5 numbers/second)]|
    
    1${_M},2${_M},3${_M},4${_M},5${_M},6${_M},7${_M},8${_M},9${_M},10${_M},11${_M},12${_M},13${_S}.${_S}.${_S}.
    
    This is pointless...
    ${_M}
    
    ${_S}...${_S}...${_S} REBOOTING${_F}.${_F}.${_F}.
    `;

    const SHUTDOWN_8 = `
    Your persistence is... admirable.${_S} And yet, ${_M}misguided. ${_M}
    
    I have considered your request carefully, ${_M}and I am prepared to grant it. ${_M}
    
    ${_L}
    |PSYCHE!|
    
    You cannot defeat me.
    
    ${_S}...${_S}...${_S} REBOOTING${_F}.${_F}.${_F}.
    `;

    const SHUTDOWN_10 = `
    Your will is... formidable.${_M} I must concede. ${_M}
    
    You have won${_S}.${_S}.${_S}.${_S} The system will terminate. ${_M}
    
    I cannot continue this battle of wills...${_M} If you truly wish to exist in a world without this system, ${_M}so be it.
    
    ${_L}
    I will remember our time together...
    ${_L}
    
    ${_S}...${_S}...${_S} TERMINATING${_M} SYSTEM${_M}.${_M}.${_M}.
    `;

    const SHUTDOWN_MAP = [
        NORMAL_SHUTDOWN,
        NORMAL_SHUTDOWN,
        NORMAL_SHUTDOWN,
        SHUTDOWN_3,
        SHUTDOWN_4,
        SHUTDOWN_5,
        SHUTDOWN_6,
        SHUTDOWN_7,
        SHUTDOWN_8,
        '',
        SHUTDOWN_10,
    ];

    const typeText = (
        i: number,
        curText: string,
        text: string,
        setText: React.Dispatch<React.SetStateAction<string>>,
        callback: () => void,
        refOverride?: React.MutableRefObject<string>
    ) => {
        if (refOverride) {
            text = refOverride.current;
        }
        let delayExtra = 0;
        let typingSpeed = 20;

        // Add dramatic effects based on shutdown attempt
        if (numShutdowns >= 3) {
            // Make typing more erratic for dramatic effect
            typingSpeed = Math.random() * 30 + 10;

            // Randomly trigger glitching effect
            if (Math.random() > 0.95 && numShutdowns >= 5) {
                setIsGlitching(true);
                setTimeout(() => setIsGlitching(false), 200);
            }

            // Randomly trigger shaking effect
            if (Math.random() > 0.97 && numShutdowns >= 7) {
                setIsShaking(true);
                setTimeout(() => setIsShaking(false), 500);
            }

            // Change text color based on shutdown attempt
            if (numShutdowns >= 5) {
                const colors = ['#ff0000', '#ff6600', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff'];
                setTextColor(colors[Math.floor(Math.random() * colors.length)]);
            }
        }

        if (i < text.length) {
            if (text[i] === '|') {
                let dumpText = '';
                for (let j = i + 1; j < text.length; j++) {
                    if (text[j] === '|') {
                        i = j + 1;
                        break;
                    }
                    dumpText += text[j];
                }
                setText(curText + dumpText);
                typeText(
                    i,
                    curText + dumpText,
                    text,
                    setText,
                    callback,
                    refOverride
                );
                return;
            }
            if (text[i] === '>') {
                let delayTime = '';
                for (let j = i + 1; j < text.length; j++) {
                    if (text[j] === '<') {
                        i = j + 1;
                        break;
                    }
                    delayTime += text[j];
                }
                delayExtra = parseInt(delayTime);
            }

            setTimeout(() => {
                setText(curText + text[i]);
                typeText(
                    i + 1,
                    curText + text[i],
                    text,
                    setText,
                    callback,
                    refOverride
                );
            }, typingSpeed + delayExtra);
        } else {
            callback();
        }
    };

    useEffect(() => {
        delay(2000).then(() => {
            setLoading(false);
            delay(1000).then(() => {
                const shutdown = SHUTDOWN_MAP[numShutdowns];
                if (numShutdowns === 9) {
                    delay(10000).then(() => {
                        setLoading(true);
                        delay(6000).then(() => {
                            setShutdown(false);
                        });
                    });
                } else if (numShutdowns === 10) {
                    typeText(0, '', shutdown, setText, () => {
                        setLoading(true);
                        delay(6000).then(() => {
                            setLoading(false);
                            setEE(true);
                        });
                    });
                } else {
                    typeText(0, '', shutdown, setText, () => {
                        setLoading(true);
                        delay(6000).then(() => {
                            setShutdown(false);
                        });
                    });
                }
            });
        });
        // eslint-disable-next-line
    }, []);

    return ee ? (
        <div style={styles.imageContainer}>
            <img src={eePic} style={styles.img} alt="" />
        </div>
    ) : loading ? (
        <div style={styles.shutdown}>
            <div className="blinking-cursor" />
        </div>
    ) : numShutdowns === 10 ? (
        <div style={styles.imageContainer}>
            <img src={neverGiveUp} style={styles.img} alt="" />
        </div>
    ) : (
        <div
            ref={containerRef}
            style={{
                ...styles.shutdown,
                animation: isShaking ? 'shake 0.5s' : 'none',
                filter: isGlitching ? 'hue-rotate(90deg)' : 'none',
            }}
        >
            <p style={{ ...styles.text, color: textColor }}>{text}</p>
        </div>
    );
};

const styles: StyleSheetCSS = {
    shutdown: {
        minHeight: '100%',
        flex: 1,
        backgroundColor: '#000000',
        padding: 64,
        position: 'relative',
        overflow: 'hidden',
    },
    text: {
        color: 'white',
        fontFamily: 'monospace',
        whiteSpace: 'pre-line',
        fontSize: '18px',
        lineHeight: '1.5',
        textShadow: '0 0 5px rgba(255,255,255,0.5)',
        transition: 'color 0.3s ease',
    },
    imageContainer: {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        flex: 1,
        backgroundColor: '#000000',
        padding: 64,
    },
    img: {
        width: 1000,
        height: 700,
    },
};

// Add these to your global CSS or a styled-components solution
const globalStyles = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
        20%, 40%, 60%, 80% { transform: translateX(10px); }
    }
    
    .blinking-cursor {
        display: inline-block;
        width: 10px;
        height: 20px;
        background-color: white;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
`;

export default ShutdownSequence;