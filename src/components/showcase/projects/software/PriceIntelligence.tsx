import React from 'react';

const PriceIntelligence: React.FC = () => {
    return (
        <div className="text-block">
            <h2>Multimodal Price Intelligence System</h2>
            <br />
            <p>
                Developed an advanced e-commerce price prediction system as part of the winning team at <b>AceCloud X RTDS Hackathon '25</b>. The system uses multimodal inputs (images and text) to predict optimal pricing for products, combining state-of-the-art computer vision and NLP techniques to achieve <b>16.48% SMAPE</b> with robust generalization.
            </p>
            <br />
            <p>
                Our team won the hackathon by building a scalable system that processes <b>100,000 products</b> with <b>9,370 engineered features</b> from three modalities: textual descriptions, semantic visual features (ResNet50), and vision-language model outputs (VLM). The challenge required handling massive-scale data processing while maintaining high prediction accuracy and computational efficiency.
            </p>
            <br />

            <h3>System Architecture:</h3>

            <style dangerouslySetInnerHTML={{
                __html: `
        @keyframes flowDown {
            0% { opacity: 0.3; transform: translateY(-8px); }
            50% { opacity: 1; }
            100% { opacity: 0.3; transform: translateY(8px); }
        }
        .architecture-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 20px 0;
        }
        .flow-node {
            border: 2px dashed #333;
            padding: 15px 25px;
            margin: 10px 0;
            text-align: center;
            min-width: 300px;
        }
        .flow-arrow {
            animation: flowDown 2s ease-in-out infinite;
            font-size: 24px;
            margin: 5px 0;
        }
        .pipeline-group {
            border: 2px dashed #666;
            padding: 15px;
            margin: 10px 0;
            text-align: left;
        }
    `}} />

            <div className="architecture-container">
                <div className="flow-node">
                    <p><b>Data Input</b></p>
                    <p>(100K products)</p>
                </div>

                <div className="flow-arrow">↓</div>

                <div className="flow-node pipeline-group">
                    <p><b>Distributed Processing</b></p>
                    <p>(42 nodes, 2TB RAM)</p>
                    <br />
                    <div style={{ paddingLeft: '10px' }}>
                        <p>├─→ <b>Text Pipeline</b> → 5,000 features (38.2%)</p>
                        <p>├─→ <b>Vision Pipeline</b> (ResNet50) → 2,048 features (32.8%)</p>
                        <p>└─→ <b>VLM Pipeline</b> (smolvlm) → 1,500 features (18.1%)</p>
                    </div>
                </div>

                <div className="flow-arrow" style={{ animationDelay: '0.4s' }}>↓</div>

                <div className="flow-node">
                    <p><b>Feature Fusion</b></p>
                    <p>(9,370 features, 98.56% sparse)</p>
                </div>

                <div className="flow-arrow" style={{ animationDelay: '0.8s' }}>↓</div>

                <div className="flow-node">
                    <p><b>LightGBM Model</b></p>
                    <p>(GPU, 20-fold CV)</p>
                </div>

                <div className="flow-arrow" style={{ animationDelay: '1.2s' }}>↓</div>

                <div className="flow-node">
                    <p><b>Price Prediction</b></p>
                    <p>SMAPE: 16.48% | R²: 0.785</p>
                </div>

                <div style={{ marginTop: '20px', textAlign: 'center' }}>
                    <p><b>Performance:</b> 48× speedup (12hrs → 15min) | Top 15% ranking</p>
                </div>
            </div>
            <br />

            <h3>Key Features:</h3>
            <ul>
                <li>
                    <p>
                        <b>Dual-Path Pipeline Architecture:</b> Implemented separate pipelines for image features (ResNet50 extracting 2,048-dimensional embeddings + local VLM extracting structured text) and catalog text (TF-IDF with 5,000 features), achieving 32.8% vision contribution and 38.2% text contribution to predictions.
                    </p>
                </li>
                <li>
                    <p>
                        <b>Vision Language Model Integration:</b> Deployed smolvlm-256m-instruct locally to extract brand names, quantities, and specifications visible in product images but missing from catalog text, generating 1,500 additional features and contributing 18.1% to model performance.
                    </p>
                </li>
                <li>
                    <p>
                        <b>Sophisticated Ensemble Models:</b> Optimized LightGBM with GPU acceleration achieving superior performance (16.48% SMAPE vs 16.72% XGBoost, 17.01% CatBoost) through 20-fold stratified cross-validation with only 0.21% standard deviation across folds.
                    </p>
                </li>
                <li>
                    <p>
                        <b>Distributed Computing with Ray:</b> Architected a Ray-based distributed cluster connecting <b>42 lab workstations</b> (1 master + 41 workers), achieving unified memory pool of <b>~2TB RAM</b> and 240+ CPU cores for parallel processing.
                    </p>
                </li>
                <li>
                    <p>
                        <b>Massive Performance Optimization:</b> Reduced total pipeline runtime from <b>12 hours to 15 minutes</b> (48× speedup) through distributed computing: image downloads (45min→3min), ResNet50 extraction (180min→8min), VLM inference (375min→12min), achieving rapid iteration cycles.
                    </p>
                </li>
                <li>
                    <p>
                        <b>Fault Tolerance & Orchestration:</b> Implemented Ray's orchestration with checkpointing every 20 images for fault recovery, parallel hyperparameter search, and graceful degradation handling 1.5% image download failures and 0.55% VLM extraction failures.
                    </p>
                </li>
                <li>
                    <p>
                        <b>Advanced Feature Engineering:</b> Extracted 9,370 features including regex-based numerical extraction (weight, volume, count), one-hot encoded categories (812 dimensions), and sparse matrix optimization (98.56% sparsity) for memory efficiency.
                    </p>
                </li>
            </ul>
            <br />

            <h3>Technical Achievements:</h3>
            <ul>
                <li><p><b>Model Performance:</b> 16.48% SMAPE, R² = 0.785 (explains 78.5% price variance), MAE = $2.34</p></li>
                <li><p><b>Ranking:</b> Top 15% on ML Challenge 2025 leaderboard</p></li>
                <li><p><b>Stability:</b> 0.21% cross-validation standard deviation demonstrating robust generalization</p></li>
                <li><p><b>Scalability:</b> Processed 75,000 training samples with 9,370-dimensional feature space</p></li>
                <li><p><b>Efficiency:</b> 48× speedup through distributed computing architecture</p></li>
            </ul>
            <br />

            <h3>Technologies Used:</h3>
            <p>
                <b>ML/DL:</b> PyTorch, LightGBM, XGBoost, CatBoost, scikit-learn, ResNet50
                <br />
                <b>NLP:</b> NLTK, TF-IDF, Regular Expressions, BeautifulSoup
                <br />
                <b>Computer Vision:</b> ResNet50, Vision Language Models (smolvlm-256m-instruct)
                <br />
                <b>Distributed Computing:</b> Ray, Dask
                <br />
                <b>Data Processing:</b> Pandas, NumPy, SciPy (sparse matrices)
                <br />
                <b>Deployment:</b> Docker, FastAPI, LM Studio (local VLM)
            </p>
            <br />

            <h3>Links:</h3>
            <ul>
                <li>
                    <a
                        rel="noreferrer"
                        target="_blank"
                        href="https://github.com/akshit7093/ML-Challenge-2025-Smart-Product-Pricing-Solution"
                    >
                        <p>
                            <b>[GitHub Repository]</b> - View Source Code & Documentation
                        </p>
                    </a>
                </li>
            </ul>
        </div>
    );
};

export default PriceIntelligence;
