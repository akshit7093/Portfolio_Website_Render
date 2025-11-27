import React from 'react';
import ghIcon from '../../assets/pictures/contact-gh.png';
import inIcon from '../../assets/pictures/contact-in.png';
import ResumeDownload from './ResumeDownload';

export interface ContactProps {}

interface SocialBoxProps {
    icon: string;
    link: string;
    label: string;
}

const SocialBox: React.FC<SocialBoxProps> = ({ link, icon, label }) => {
    return (
        <a rel="noreferrer" target="_blank" href={link}>
            <div className="big-button-container" style={styles.social}>
                <img src={icon} alt={label} style={styles.socialImage} />
            </div>
        </a>
    );
};

const Contact: React.FC<ContactProps> = (props) => {
    return (
        <div className="site-page-content">
            <div style={styles.header}>
                <h1>Contact</h1>
                <div style={styles.socials}>
                    <SocialBox
                        icon={ghIcon}
                        link={'https://github.com/akshit7093'}
                        label="GitHub"
                    />
                    <SocialBox
                        icon={inIcon}
                        link={'https://www.linkedin.com/in/akshit-sharma-475a94271/'}
                        label="LinkedIn"
                    />
                </div>
            </div>
            <div className="text-block">
                <p>
                    I'm actively seeking opportunities in Backend Development, AI/ML Engineering, and Cloud-Native Systems. Feel free to reach out if you have any opportunities or would like to collaborate!
                </p>
                <br />
                <p>
                    <b>Email: </b>
                    <a href="mailto:akshitsharma7096@gmail.com">
                        akshitsharma7096@gmail.com
                    </a>
                </p>
                <p>
                    <b>Phone: </b>
                    <a href="tel:+918810248097">
                        +91 8810248097
                    </a>
                </p>
                <br />
                <div style={styles.linksSection}>
                    <h3>Find me on:</h3>
                    <br />
                    <ul>
                        <li>
                            <p>
                                <b>GitHub:</b>{' '}
                                <a
                                    rel="noreferrer"
                                    target="_blank"
                                    href="https://github.com/akshit7093"
                                >
                                    github.com/akshit7093
                                </a>
                            </p>
                        </li>
                        <li>
                            <p>
                                <b>LinkedIn:</b>{' '}
                                <a
                                    rel="noreferrer"
                                    target="_blank"
                                    href="https://www.linkedin.com/in/akshit-sharma-475a94271/"
                                >
                                    linkedin.com/in/akshit-sharma-475a94271
                                </a>
                            </p>
                        </li>
                        <li>
                            <p>
                                <b>LeetCode:</b>{' '}
                                <a
                                    rel="noreferrer"
                                    target="_blank"
                                    href="https://leetcode.com/u/akshitsharma7093/"
                                >
                                    leetcode.com/u/akshitsharma7093
                                </a>
                            </p>
                        </li>
                        <li>
                            <p>
                                <b>CodeForces:</b>{' '}
                                <a
                                    rel="noreferrer"
                                    target="_blank"
                                    href="https://codeforces.com/profile/akshit7093"
                                >
                                    codeforces.com/profile/akshit7093
                                </a>
                            </p>
                        </li>
                    </ul>
                </div>
            </div>
            <ResumeDownload altText="Need a copy of my Resume?" />
        </div>
    );
};

const styles: StyleSheetCSS = {
    socialImage: {
        width: 36,
        height: 36,
    },
    header: {
        alignItems: 'flex-end',
        justifyContent: 'space-between',
    },
    socials: {
        marginBottom: 16,
        justifyContent: 'flex-end',
    },
    social: {
        width: 4,
        height: 4,
        justifyContent: 'center',
        alignItems: 'center',
        marginLeft: 8,
    },
    linksSection: {
        marginTop: 16,
    },
};

export default Contact;
