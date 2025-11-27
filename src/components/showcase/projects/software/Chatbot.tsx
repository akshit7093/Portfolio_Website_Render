import React from 'react';

const Chatbot: React.FC = () => {
    return (
        <div style={{ padding: '24px', height: '100%', boxSizing: 'border-box', overflowY: 'auto' }}>
            <h2>Universal Chatbot Platform</h2>
            <h3>RAG & Fine-tuned LLMs</h3>
            <br />
            <div className="text-block">
                <p>
                    Built during my internship at Akanila Technologies, this platform enables businesses to deploy intelligent chatbots with minimal configuration. The system leverages fine-tuned LLMs and RAG for context-aware responses.
                </p>
                <br />
                <h3>Key Features:</h3>
                <ul>
                    <li>
                        <p>
                            Fine-tuned Llama3.1 using LoRA for efficient adaptation to specific domains
                        </p>
                    </li>
                    <li>
                        <p>
                            Integrated FAISS vector database for fast similarity search and retrieval
                        </p>
                    </li>
                    <li>
                        <p>
                            Modular FastAPI backend with 65% code reusability
                        </p>
                    </li>
                    <li>
                        <p>
                            Automated CI/CD pipeline with Docker and Kubernetes on AWS EC2
                        </p>
                    </li>
                </ul>
                <br />
                <h3>Technologies Used:</h3>
                <p>
                    Python, Llama3.1, LoRA, FAISS, FastAPI, Docker, Kubernetes, AWS EC2
                </p>
                <br />
                <h3>Links:</h3>
                <ul>
                    <li>
                        <a
                            rel="noreferrer"
                            target="_blank"
                            href="https://github.com/akshit7093/Chatbot-for-websites"
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

export default Chatbot;
