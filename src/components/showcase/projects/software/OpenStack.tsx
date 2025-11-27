import React from 'react';

const OpenStack: React.FC = () => {
    return (
        <div style={{ padding: '24px', height: '100%', boxSizing: 'border-box', overflowY: 'auto' }}>
            <h2>OpenStack Cloud Manager</h2>
            <h3>Natural Language Interface</h3>
            <br />
            <div className="text-block">
                <p>
                    Built a sophisticated cloud management system that interfaces with OpenStack infrastructure APIs, enabling users to manage cloud resources using natural language commands.
                </p>
                <br />
                <p>
                    The system allows users to issue natural language prompts such as "create a server" or "delete a volume", which an AI agent translates into precise OpenStack API calls. This dramatically simplifies cloud infrastructure management and makes it accessible to users without deep technical knowledge of OpenStack commands.
                </p>
                <br />
                <h3>Key Features:</h3>
                <ul>
                    <li>
                        <p>
                            <b>AI-Powered Natural Language Interface:</b> Integrated LangChain with Google's Gemini-2.5 pro model to parse and understand user intent from natural language commands.
                        </p>
                    </li>
                    <li>
                        <p>
                            <b>OpenStack API Integration:</b> Comprehensive integration with OpenStack SDK for VM creation, volume management, network configuration, and resource monitoring.
                        </p>
                    </li>
                    <li>
                        <p>
                            <b>Interactive CLI & Web App:</b> Built both a command-line interface and a web application for remote management, providing flexibility for different use cases.
                        </p>
                    </li>
                    <li>
                        <p>
                            <b>Resource Analytics:</b> Implemented resource analytics and container monitoring per VM, providing real-time insights into infrastructure usage.
                        </p>
                    </li>
                    <li>
                        <p>
                            <b>RESTful Backend:</b> Designed a scalable RESTful backend using FastAPI with proper error handling and authentication.
                        </p>
                    </li>
                    <li>
                        <p>
                            <b>Containerized Deployment:</b> Containerized the entire application using Docker for easy deployment and portability.
                        </p>
                    </li>
                </ul>
                <br />
                <h3>Technologies Used:</h3>
                <p>
                    Python, OpenStack SDK, Google Gemini-2.5 Pro, LangChain, FastAPI, Docker
                </p>
                <br />
                <h3>Links:</h3>
                <ul>
                    <li>
                        <a
                            rel="noreferrer"
                            target="_blank"
                            href="https://github.com/akshit7093/VM_manager_AgenticAi"
                        >
                            <p>
                                <b>[GitHub Repository]</b> - View Source Code
                            </p>
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    );
};

export default OpenStack;
