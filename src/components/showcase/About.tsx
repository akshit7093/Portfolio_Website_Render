import React from 'react';
import { Link } from 'react-router-dom';
import ResumeDownload from './ResumeDownload';

export interface AboutProps {}

const About: React.FC<AboutProps> = (props) => {
    return (
        <div className="site-page-content">
            <h1 style={{ marginLeft: -16 }}>Welcome</h1>
            <h3>I'm Akshit Sharma</h3>
            <br />
            <div className="text-block">
                <p>
                    I'm a final-year B.Tech student specializing in Artificial Intelligence and Data Science at Maharaja Agrasen Institute of Technology. With a CGPA of 8.96/10, I'm passionate about building scalable backend systems and intelligent AI/ML solutions.
                </p>
                <br />
                <p>
                    Thank you for taking the time to check out my portfolio. I really hope you enjoy exploring it. If you have any questions or comments, feel free to contact me using{' '}
                    <Link to="/contact">this page</Link> or shoot me an email at{' '}
                    <a href="mailto:akshitsharma7096@gmail.com">
                        akshitsharma7096@gmail.com
                    </a>
                </p>
            </div>
            <ResumeDownload />
            <div className="text-block">
                <h3>About Me</h3>
                <br />
                <p>
                    From a young age, I've been fascinated by technology and problem-solving. This curiosity naturally led me to pursue Computer Science with a specialization in Artificial Intelligence and Data Science. My journey in tech has been driven by a passion for understanding how intelligent systems work and building solutions that can make a real impact.
                </p>
                <br />
                <p>
                    During my academic journey, I've worked on cutting-edge projects involving cloud infrastructure, natural language processing, and distributed systems. I'm particularly interested in the intersection of AI and cloud-native technologies, where I can leverage my skills in both domains to build scalable, intelligent applications.
                </p>
                <br />
                <p>
                    Currently, I'm working as a Research Intern at the Directorate of Research, Government of Arunachal Pradesh, where I'm developing a low-resource speech-to-speech translation pipeline for endangered languages. This project combines my interests in NLP, deep learning, and social impact.
                </p>
                <br />
                <h3>My Technical Journey</h3>
                <br />
                <p>
                    My professional experience includes developing a universal chatbot platform at Akanila Technologies, where I fine-tuned Llama3.1 using LoRA and integrated RAG with FAISS. I designed a flexible Python backend with FastAPI and implemented automated deployments on AWS EC2 using Docker and Kubernetes.
                </p>
                <br />
                <p>
                    At CodSoft, I worked on machine learning projects including a credit card fraud detection system using XGBoost, where I engineered features from behavioral and time-series data to significantly reduce false positives while maintaining high recall.
                </p>
                <br />
                <h3>Beyond Code</h3>
                <br />
                <p>
                    When I'm not coding, I enjoy solving algorithmic challenges on LeetCode and CodeForces, which helps me sharpen my problem-solving skills. I'm also an active participant in hackathons and recently won the AceCloud X RTDS Hackathon '25, where my team built a multimodal price intelligence system using distributed computing.
                </p>
                <br />
                <p>
                    I believe in continuous learning and regularly work on expanding my skill set through online courses and certifications. I've completed specializations in Machine Learning, Deep Learning, and Generative AI from platforms like Coursera, Google, and Pwskills.
                </p>
                <br />
                <p>
                    Thanks for reading about me! I hope you enjoy exploring the rest of my portfolio and the projects I've worked on. Feel free to reach me through the{' '}
                    <Link to="/contact">contact page</Link> or email me at{' '}
                    <a href="mailto:akshitsharma7096@gmail.com">
                        akshitsharma7096@gmail.com
                    </a>
                </p>
            </div>
        </div>
    );
};

export default About;
