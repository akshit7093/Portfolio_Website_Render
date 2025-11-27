import React from 'react';

export interface CertificationsProps {}

const Certifications: React.FC<CertificationsProps> = (props) => {
    return (
        <div className="site-page-content">
            <h1>Certifications</h1>
            <h3>& Continuous Learning</h3>
            <br />
            <div className="text-block">
                <p>
                    I believe in continuous learning and regularly work on expanding my skill set through structured courses and certifications. Below are some of the key certifications I've earned that demonstrate my expertise in various domains.
                </p>
            </div>
            <br />
            <div className="text-block">
                <h2>Data Science & Machine Learning</h2>
                <br />
                <ul>
                    <li>
                        <h3>Data Science Certification</h3>
                        <p><b>Provider:</b> Pwskills</p>
                        <p>Comprehensive coverage of data science fundamentals, including statistics, data analysis, visualization, and machine learning algorithms.</p>
                        <br />
                    </li>
                    <li>
                        <h3>Machine Learning and Deep Learning Specialization</h3>
                        <p><b>Provider:</b> Coursera</p>
                        <p>In-depth specialization covering supervised learning, unsupervised learning, deep neural networks, CNNs, RNNs, and modern deep learning architectures.</p>
                        <br />
                    </li>
                </ul>
            </div>
            <br />
            <div className="text-block">
                <h2>Cloud Computing & Architecture</h2>
                <br />
                <ul>
                    <li>
                        <h3>AWS Solutions Architect Virtual Experience Program</h3>
                        <p><b>Provider:</b> Forage</p>
                        <p>Practical experience in designing scalable, reliable, and cost-effective AWS architectures. Covered EC2, S3, RDS, Lambda, and other core AWS services.</p>
                        <br />
                    </li>
                </ul>
            </div>
            <br />
            <div className="text-block">
                <h2>Generative AI & Advanced Models</h2>
                <br />
                <ul>
                    <li>
                        <h3>Introduction to Generative AI</h3>
                        <p><b>Provider:</b> Google</p>
                        <p>Foundational understanding of generative AI, including GANs, VAEs, and transformer-based models.</p>
                        <br />
                    </li>
                    <li>
                        <h3>Develop GenAI Apps with Gemini and Streamlit</h3>
                        <p><b>Provider:</b> Google</p>
                        <p>Hands-on experience building generative AI applications using Google's Gemini model and Streamlit for rapid prototyping.</p>
                        <br />
                    </li>
                    <li>
                        <h3>Prompt Design in Vertex AI</h3>
                        <p><b>Provider:</b> Google</p>
                        <p>Advanced techniques for prompt engineering and optimization in Google's Vertex AI platform for better model outputs.</p>
                        <br />
                    </li>
                </ul>
            </div>
            <br />
            <div className="text-block">
                <h2>Technical Skills</h2>
                <br />
                <p>
                    Through these certifications and my academic coursework, I've developed expertise in:
                </p>
                <br />
                <ul>
                    <li><p><b>Programming Languages:</b> Python, Java, C/C++, JavaScript, SQL, TypeScript</p></li>
                    <li><p><b>Web Frameworks:</b> React, Node.js, Flask, FastAPI</p></li>
                    <li><p><b>Data Science:</b> Pandas, NumPy, Matplotlib, Scikit-learn</p></li>
                    <li><p><b>Databases:</b> MongoDB, PostgreSQL</p></li>
                    <li><p><b>ML/AI Frameworks:</b> TensorFlow, PyTorch, Transformers, LangChain</p></li>
                    <li><p><b>ML Techniques:</b> NLP, Computer Vision, RAG, Fine-tuning</p></li>
                    <li><p><b>Cloud & DevOps:</b> AWS, Google Cloud Platform, Docker, Kubernetes, OpenStack SDK</p></li>
                    <li><p><b>Systems:</b> Unix/Linux, TCP/IP Networking, Git</p></li>
                </ul>
            </div>
            <br />
            <div className="text-block">
                <h2>Academic Coursework</h2>
                <br />
                <p>
                    Relevant courses from my B.Tech program at Maharaja Agrasen Institute of Technology:
                </p>
                <br />
                <ul>
                    <li><p>Machine Learning</p></li>
                    <li><p>Data Mining</p></li>
                    <li><p>Image Processing</p></li>
                    <li><p>Data Structures and Algorithms</p></li>
                    <li><p>Computer Networks</p></li>
                </ul>
            </div>
        </div>
    );
};

export default Certifications;
