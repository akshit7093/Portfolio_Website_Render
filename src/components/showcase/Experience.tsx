import React from 'react';
import ResumeDownload from './ResumeDownload';

export interface ExperienceProps {}

const Experience: React.FC<ExperienceProps> = (props) => {
    return (
        <div className="site-page-content">
            <ResumeDownload />
            <div style={styles.headerContainer}>
                <div style={styles.header}>
                    <div style={styles.headerRow}>
                        <h1>Directorate of Research</h1>
                        <a
                            rel="noreferrer"
                            target="_blank"
                            href={'https://arunachal.gov.in/'}
                        >
                            <h4>Government of Arunachal Pradesh</h4>
                        </a>
                    </div>
                    <div style={styles.headerRow}>
                        <h3>Research Intern</h3>
                        <b>
                            <p>August 2025 - Present</p>
                        </b>
                    </div>
                </div>
            </div>
            <div className="text-block">
                <p>
                    Working on cutting-edge research in speech-to-speech translation for endangered languages with context-dependent meanings. Focusing on low-resource language processing and real-time deployment solutions.
                </p>
                <br />
                <ul>
                    <li>
                        <p>
                            Developed a low-resource speech-to-speech translation pipeline using Wav2Vec 2.0 for Automatic Speech Recognition (ASR), MarianMT for Neural Machine Translation (NMT), and Tacotron 2 for Text-to-Speech (TTS).
                        </p>
                    </li>
                    <li>
                        <p>
                            Optimized system architecture to reduce translation latency to under 2 seconds, enabling real-time deployment for field use in remote areas.
                        </p>
                    </li>
                    <li>
                        <p>
                            Implemented specialized handling for context-dependent meanings in endangered languages, improving translation accuracy and cultural preservation.
                        </p>
                    </li>
                </ul>
            </div>
            <div style={styles.headerContainer}>
                <div style={styles.header}>
                    <div style={styles.headerRow}>
                        <h1>Akanila Technologies</h1>
                        <a
                            target="_blank"
                            rel="noreferrer"
                            href={'https://github.com/akshit7093/Chatbot-for-websites'}
                        >
                            <h4>View Project</h4>
                        </a>
                    </div>
                    <div style={styles.headerRow}>
                        <h3>Deep Learning Intern</h3>
                        <b>
                            <p>July 2024 - December 2024</p>
                        </b>
                    </div>
                </div>
            </div>
            <div className="text-block">
                <p>
                    Built a universal chatbot platform leveraging state-of-the-art LLMs and retrieval-augmented generation. Designed scalable backend architecture and implemented cloud deployment using containerization and orchestration technologies.
                </p>
                <br />
                <ul>
                    <li>
                        <p>
                            Built a universal chatbot platform by fine-tuning Llama3.1 using LoRA (Low-Rank Adaptation), achieving efficient parameter-efficient training.
                        </p>
                    </li>
                    <li>
                        <p>
                            Integrated Retrieval-Augmented Generation (RAG) with FAISS vector database for enhanced information retrieval and context-aware responses.
                        </p>
                    </li>
                    <li>
                        <p>
                            Designed a flexible Python backend with modular components in FastAPI, increasing code reusability to 65% and enabling rapid feature development.
                        </p>
                    </li>
                    <li>
                        <p>
                            Implemented automated deployments on AWS EC2, leveraging Docker for containerization and Kubernetes for container orchestration, ensuring high availability and scalability.
                        </p>
                    </li>
                </ul>
            </div>
            <div style={styles.headerContainer}>
                <div style={styles.header}>
                    <div style={styles.headerRow}>
                        <h1>CodSoft</h1>
                        <a
                            target="_blank"
                            rel="noreferrer"
                            href={'https://github.com/akshit7093/CODSOFT'}
                        >
                            <h4>View Projects</h4>
                        </a>
                    </div>
                    <div style={styles.headerRow}>
                        <h3>Machine Learning Intern</h3>
                        <b>
                            <p>August 2024 - September 2024</p>
                        </b>
                    </div>
                </div>
            </div>
            <div className="text-block">
                <p>
                    Developed machine learning models for fraud detection and spam classification. Worked with large-scale datasets and implemented feature engineering techniques to improve model performance.
                </p>
                <br />
                <ul>
                    <li>
                        <p>
                            Developed a credit card fraud detection system using XGBoost, analyzing over 1 million transaction records to identify fraudulent patterns.
                        </p>
                    </li>
                    <li>
                        <p>
                            Engineered 20+ features from behavioral and time-series data, then trained an XGBoost model on AWS SageMaker to reduce false positives from 20% to 5% while maintaining recall above 90%.
                        </p>
                    </li>
                    <li>
                        <p>
                            Built an NLP model for SMS spam detection using Python and scikit-learn, achieving 95% accuracy on test data through careful feature extraction and model tuning.
                        </p>
                    </li>
                </ul>
            </div>
        </div>
    );
};

const styles: StyleSheetCSS = {
    header: {
        flexDirection: 'column',
        justifyContent: 'space-between',
        width: '100%',
    },
    skillRow: {
        flex: 1,
        justifyContent: 'space-between',
    },
    skillName: {
        minWidth: 56,
    },
    skill: {
        flex: 1,
        padding: 8,
        alignItems: 'center',
    },
    progressBar: {
        flex: 1,
        background: 'red',
        marginLeft: 8,
        height: 8,
    },
    hoverLogo: {
        height: 32,
        marginBottom: 16,
    },
    headerContainer: {
        alignItems: 'flex-end',
        width: '100%',
        justifyContent: 'center',
    },
    hoverText: {
        marginBottom: 8,
    },
    indent: {
        marginLeft: 24,
    },
    headerRow: {
        justifyContent: 'space-between',
        alignItems: 'flex-end',
    },
    row: {
        display: 'flex',
        justifyContent: 'space-between',
    },
};

export default Experience;
