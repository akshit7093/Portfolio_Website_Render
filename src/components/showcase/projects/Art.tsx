import React from 'react';

export interface AchievementsProps {}

const Achievements: React.FC<AchievementsProps> = (props) => {
    return (
        <div className="site-page-content">
            <h1>Achievements</h1>
            <h3>& Awards</h3>
            <br />
            <div className="text-block">
                <p>
                    Throughout my academic and professional journey, I've participated in various hackathons, competitions, and challenges. Here are some of my notable achievements that demonstrate my problem-solving abilities and technical expertise.
                </p>
            </div>
            <br />
            <div className="text-block">
                <h2>🏆 Winner - AceCloud X RTDS Hackathon '25</h2>
                <br />
                <p>
                    Our team won the AceCloud X RTDS Hackathon '25, a prestigious hackathon focused on cloud computing and machine learning challenges. We developed a multimodal price intelligence system that impressed the judges with its innovative approach and technical excellence.
                </p>
                <br />
                <h3>Project: Multimodal Price Intelligence System</h3>
                <br />
                <p>
                    <b>Challenge:</b> Build a scalable e-commerce price prediction system that can handle multimodal inputs (images and text) with high accuracy and low latency.
                </p>
                <br />
                <p>
                    <b>Our Solution:</b>
                </p>
                <ul>
                    <li>
                        <p>
                            Implemented dual-path pipelines for image and text processing using ResNet50 and NLP techniques
                        </p>
                    </li>
                    <li>
                        <p>
                            Built sophisticated ensemble models combining LightGBM, CatBoost, and Temporal Fusion Transformer
                        </p>
                    </li>
                    <li>
                        <p>
                            Architected a Ray-based distributed cluster connecting 40 workstations with 2TB unified memory
                        </p>
                    </li>
                    <li>
                        <p>
                            Optimized training runtime from 13 hours to 4 hours through distributed computing
                        </p>
                    </li>
                </ul>
                <br />
                <p>
                    <b>Impact:</b> The system demonstrated exceptional performance in handling large-scale multimodal data with efficient resource utilization, earning us the first place.
                </p>
            </div>
            <br />
            <div className="text-block">
                <h2>Academic Excellence</h2>
                <br />
                <h3>CGPA: 8.96/10</h3>
                <p>
                    Maintained a strong academic record throughout my B.Tech program in Computer Science with specialization in Artificial Intelligence and Data Science at Maharaja Agrasen Institute of Technology.
                </p>
                <br />
                <p>
                    <b>Duration:</b> June 2022 - June 2026 (Expected)
                </p>
            </div>
            <br />
            <div className="text-block">
                <h2>Competitive Programming</h2>
                <br />
                <p>
                    I regularly practice algorithmic problem-solving on competitive programming platforms to sharpen my skills:
                </p>
                <br />
                <ul>
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
                <br />
                <p>
                    Regular practice on these platforms has helped me develop strong problem-solving skills, understand complex algorithms, and optimize code for performance.
                </p>
            </div>
            <br />
            <div className="text-block">
                <h2>Research & Publications</h2>
                <br />
                <p>
                    Currently working as a Research Intern at the Directorate of Research, Government of Arunachal Pradesh, focusing on low-resource speech-to-speech translation for endangered languages. This work has potential for publication and real-world impact in preserving linguistic diversity.
                </p>
            </div>
            <br />
            <div className="text-block">
                <h2>Open Source Contributions</h2>
                <br />
                <p>
                    I actively contribute to open-source projects and maintain several repositories on GitHub:
                </p>
                <br />
                <ul>
                    <li>
                        <p>
                            <a
                                rel="noreferrer"
                                target="_blank"
                                href="https://github.com/akshit7093/VM_manager_AgenticAi"
                            >
                                OpenStack Cloud Management System
                            </a> - AI-powered cloud infrastructure management
                        </p>
                    </li>
                    <li>
                        <p>
                            <a
                                rel="noreferrer"
                                target="_blank"
                                href="https://github.com/akshit7093/Chatbot-for-websites"
                            >
                                Universal Chatbot Platform
                            </a> - LLM-based chatbot with RAG
                        </p>
                    </li>
                    <li>
                        <p>
                            <a
                                rel="noreferrer"
                                target="_blank"
                                href="https://github.com/akshit7093/CODSOFT"
                            >
                                ML Projects Collection
                            </a> - Fraud detection and spam classification models
                        </p>
                    </li>
                </ul>
                <br />
                <p>
                    Check out my{' '}
                    <a
                        rel="noreferrer"
                        target="_blank"
                        href="https://github.com/akshit7093"
                    >
                        GitHub profile
                    </a>{' '}
                    for more projects and contributions.
                </p>
            </div>
        </div>
    );
};

export default Achievements;
