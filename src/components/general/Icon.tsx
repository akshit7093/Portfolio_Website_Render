import React from 'react';
import getIconByName, { IconName } from '../../assets/icons';

export interface IconProps {
    style?: React.CSSProperties;
    icon: IconName;
    size?: number;
}

const Icon: React.FC<IconProps> = ({ icon, style, size }) => {
    const iconStyle = Object.assign(
        {},
        styles.icon,
        style,
        size && { width: size, height: size }
    );
    const src = getIconByName(icon);
    if (!src) return null;

    return (
        <img
            style={iconStyle}
            alt={''}
            src={src as unknown as string}
        />
    );
};

const styles: StyleSheetCSS = {
    icon: {
        imageRendering: 'pixelated',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        msUserSelect: 'none',
        pointerEvents: 'none',
    },
};

export default Icon;
