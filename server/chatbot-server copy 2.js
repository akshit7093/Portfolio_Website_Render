const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const chalk = require('chalk');
const { marked } = require('marked');
const TerminalRenderer = require('marked-terminal');
const boxen = require('boxen');
const gradient = require('gradient-string');
require('dotenv').config();
const fetch = require('node-fetch');

const PORT = process.env.CHATBOT_PORT || 4000;
const ORIGINS = (process.env.ALLOWED_ORIGINS || 'http://localhost:3000,http://127.0.0.1:3000').split(',');

const app = express();
app.use(cors({ origin: ORIGINS }));

const server = app.listen(PORT, () => {
  console.log(`[chatbot-server] Listening on http://localhost:${PORT}`);
});

const wss = new WebSocket.Server({ server });

// Configure marked to use terminal renderer
marked.setOptions({
  renderer: new TerminalRenderer({
    code: chalk.yellow,
    blockquote: chalk.gray.italic,
    html: chalk.gray,
    heading: chalk.green.bold,
    firstHeading: chalk.magenta.underline.bold,
    hr: chalk.reset,
    listitem: chalk.cyan,
    list: (body) => chalk.cyan(body),
    paragraph: chalk.white,
    strong: chalk.bold.cyan,
    em: chalk.italic.yellow,
    codespan: chalk.yellow.dim
  })
});

class AkshitChatbot {
  constructor(outputCallback) {
    this.output = outputCallback;
    this.proxyUrl = this.resolveProxyUrl();
    this.conversationHistory = [];

    if (!this.proxyUrl || !this.isValidUrl(this.proxyUrl)) {
      this.output(chalk.red.bold('❌ CONFIGURATION ERROR: Backend URL not available\n'));
    }
  }

  resolveProxyUrl() {
    const envSources = [
      process.env.AKSHIT_CHATBOT_PROXY_URL,
      process.env.CHATBOT_PROXY_URL,
      process.env.VERCEL_URL,
      process.env.API_BASE_URL,
      process.env.BACKEND_URL
    ];

    for (const url of envSources) {
      if (url && this.isValidUrl(url)) {
        return this.normalizeApiUrl(url);
      }
    }
    return 'https://akshit-cli-backend-dzp6bnms7-jokers-projects-741f992f.vercel.app/api/chat';
  }

  maskUrl(url) {
    if (!url) return 'Not configured';
    try {
      const urlObj = new URL(url);
      const domain = urlObj.hostname;
      const maskedDomain = domain.length > 20
        ? domain.substring(0, 8) + '...' + domain.substring(domain.length - 8)
        : domain;
      return `${urlObj.protocol}//${maskedDomain}${urlObj.pathname}`;
    } catch {
      return 'Invalid URL format';
    }
  }

  normalizeApiUrl(url) {
    if (!url) return url;
    url = url.replace(/\/$/, '');
    if (!url.endsWith('/api/chat')) {
      if (url.endsWith('/api')) {
        url += '/chat';
      } else {
        url += '/api/chat';
      }
    }
    return url;
  }

  isValidUrl(url) {
    if (!url || typeof url !== 'string') return false;
    if (url.toLowerCase() === 'undefined' || url.toLowerCase() === 'null' || url.toLowerCase() === 'localhost' || url === '') return false;
    try {
      const urlObj = new URL(url);
      return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
    } catch (error) {
      return false;
    }
  }

  async generateResponse(userInput) {
    try {
      const payload = this.buildPayload(userInput);
      const response = await fetch(this.proxyUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Access-Token': 'akshit-portfolio-cli-v1'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        if (response.status === 429) throw new Error('Rate limit exceeded. Please wait a minute before trying again.');
        if (response.status === 401) throw new Error('Security verification failed.');
        const errorText = await response.text();
        throw new Error(`API request failed with status ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      return {
        response: "I'm sorry, I'm having trouble connecting to my knowledge base. Please verify the backend service is running and properly configured.",
        action: 'error'
      };
    }
  }

  detectAction(input) {
    const lowerInput = input.toLowerCase();
    if (lowerInput.includes('linkedin') || lowerInput.includes('professional profile') || lowerInput.includes('profile update') || lowerInput.includes('professional background')) return 'linkedin';
    if (lowerInput.includes('github') || lowerInput.includes('repository') || lowerInput.includes('repo') || lowerInput.includes('projects') || lowerInput.includes('code')) return 'github';
    if ((lowerInput.includes('job') || lowerInput.includes('position') || lowerInput.includes('role')) && (lowerInput.includes('fit') || lowerInput.includes('match') || lowerInput.includes('analysis') || lowerInput.includes('suitable') || lowerInput.includes('http'))) return 'job-analysis';
    if (lowerInput.includes('contact') || lowerInput.includes('connect') || lowerInput.includes('reach out') || lowerInput.includes('email') || lowerInput.includes('message')) return 'contact';
    if (lowerInput.includes('blog') || lowerInput.includes('website') || lowerInput.includes('webpage') || lowerInput.includes('latest updates') || lowerInput.includes('recent posts')) return 'webpage';
    if (lowerInput.includes('resume') || lowerInput.includes('cv') || lowerInput.includes('education') || lowerInput.includes('experience')) return 'resume';
    return 'chat';
  }

  buildPayload(userInput) {
    const context = this.conversationHistory.slice(-3).map(msg => `${msg.role}: ${msg.content}`).join('\n');
    const contextualPrompt = `You are an AI assistant representing Akshit Sharma, an AI/ML Engineering student with comprehensive knowledge about his professional profile.

EDUCATION:
- B.Tech Computer Science (AI & Data Science Specialization)
- Maharaja Agrasen Institute of Technology
- CGPA: 8.96/10 (2022-2026)

RECENT ACHIEVEMENTS:
- Winner of AceCloud X RTDS Hackathon '25
- Developed multiple high-accuracy ML models (89-95% accuracy)

TECHNICAL SKILLS:
- Programming: Python (Advanced), C/C++, Java, JavaScript, SQL
- AI/ML: TensorFlow, PyTorch, BERT, Transformers, NLP, Computer Vision, RAG
- Cloud: Google Cloud Platform, AWS, OpenStack SDK
- Tools: Git, Linux, MongoDB, Docker, Kubernetes, Flask, React

KEY PROJECTS:
1. OpenStack Cloud Management System (87% accuracy, 300ms response time)
2. SignEase ASL Video Platform (89% accuracy, optimized latency)
3. Universal Website Chatbot (90% accuracy using fine-tuned Llama 3.1)

WORK EXPERIENCE:
- ML Intern at CodSoft (Aug-Sep 2024): Movie recommendation (92% accuracy)
- ML Intern at Cantilever.in (Jul-Aug 2024): BERT sentiment analysis (88% accuracy)

CONTACT:
- Email: akshitsharma7096@gmail.com
- Phone: +91 8810248097
- GitHub: https://github.com/akshit7093
- LinkedIn: https://linkedin.com/in/akshitsharma
- LeetCode: https://leetcode.com/akshitsharma

Previous conversation:
${context}

User: ${userInput}
Assistant:`;

    return { contextualPrompt };
  }

  formatResponse(data) {
    try {
      let formattedResponse = data.response; // Keep raw markdown

      if (data.action === 'github' && data.repositories) {
        formattedResponse += `\n\n**📊 Analyzed ${data.repositories.length} repositories • Total: ${data.totalRepos}**`;
      }

      if (data.action === 'linkedin' && data.profileData) {
        formattedResponse += `\n\n**💼 Data source: LinkedIn profile • Last updated: ${data.profileData.lastUpdated}**`;
      }

      if (data.action === 'job-analysis' && data.jobUrl) {
        formattedResponse += `\n\n**🔗 Analyzed job posting: ${data.jobUrl}**`;
      }

      if (data.action === 'contact' && data.actionUrl) {
        formattedResponse += `\n\n**🔗 Quick Action: ${data.actionUrl}**`;
      }

      if (data.action === 'resume' && data.resumeData) {
        formattedResponse += `\n\n**📄 Source: Comprehensive resume data • Last updated: ${data.resumeData.lastUpdated}**`;
      }

      return formattedResponse;
    } catch (error) {
      return data.response;
    }
  }

  getActionColor(action) {
    const colors = { 'chat': 'blue', 'linkedin': 'cyan', 'github': 'green', 'job-analysis': 'yellow', 'contact': 'magenta', 'webpage': 'blue', 'resume': 'red', 'error': 'red' };
    return colors[action] || 'blue';
  }

  async handleInput(input) {
    const userInput = input.trim();
    if (!userInput) return;

    if (userInput.toLowerCase() === 'exit' || userInput.toLowerCase() === 'quit') {
      this.output(boxen(chalk.yellow.bold('👋 Thanks for exploring Akshit\'s Enhanced AI Assistant!'), { padding: 1, borderStyle: 'double', borderColor: 'yellow' }));
      return;
    }

    if (userInput === 'help') {
      this.output(boxen(chalk.cyan.bold('🔧 Available Commands:\n\n') + chalk.white('• linkedin, github, job analysis, contact, resume\n'), { padding: 1, borderStyle: 'round', borderColor: 'cyan' }));
      return;
    }

    this.conversationHistory.push({ role: 'user', content: userInput });

    const frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const actionName = this.detectAction(userInput);
    const actionEmojis = { 'linkedin': '💼', 'github': '📊', 'job-analysis': '🎯', 'contact': '📱', 'webpage': '🌐', 'resume': '📄', 'chat': '🤖' };

    let i = 0;
    const thinking = setInterval(() => {
      this.output(`\r${chalk.blue.bold('🤖 Assistant:')} ${chalk.gray(frames[i % frames.length] + ' processing ' + actionEmojis[actionName] + ' ' + actionName + '...')}`);
      i++;
    }, 100);

    const responseData = await this.generateResponse(userInput);

    clearInterval(thinking);
    // Clear the thinking line
    // We send a carriage return and spaces to clear, then carriage return again
    // But since we are sending to WebSocket, we just send the final output. 
    // The frontend handles \r to replace the last line.
    // So we just send the final response now.

    // Actually, to clear the spinner on frontend, we can send a \r with empty string or just the new content?
    // The frontend logic replaces the line if it has \r.
    // Let's just send the final output. The frontend will append it.
    // Wait, if I don't send \r, the spinner line stays there?
    // The frontend logic: if text includes \r, it splits and updates the last line.
    // So if I send the final response, it will be appended. The spinner line will remain as the "last line" before this new one?
    // No, the spinner update replaces the last line.
    // So if I want to "finalize" it, I should send the final content.
    // But the spinner was sent as `\r...`.
    // If I send `\r ` (empty) it might clear it.
    // Let's just send the formatted response. The user will see the spinner stop updating, and then the response appear.
    // Ideally I should replace the spinner line with the response header.

    this.output(`\r${chalk.blue.bold('\n🤖 Assistant:')}\n`);
    this.output(this.formatResponse(responseData));

    this.conversationHistory.push({ role: 'assistant', content: responseData.response });
    if (this.conversationHistory.length > 20) this.conversationHistory = this.conversationHistory.slice(-16);
  }

  start() {
    const title = gradient.rainbow.multiline(`
╔══════════════════════════════════════════════════╗
║         🤖 AKSHIT'S ENHANCED AI ASSISTANT       ║
║           Powered by Advanced Integrations       ║
╚══════════════════════════════════════════════════╝`);
    this.output('\n' + title + '\n');

    this.output(boxen(
      chalk.yellow.bold('🚀 Enhanced Features Available:\n\n') +
      chalk.white('💼 **LinkedIn Integration**\n') + chalk.gray('  • "linkedin profile updates"\n') +
      chalk.white('📊 **GitHub Analysis**\n') + chalk.gray('  • "show github repositories"\n') +
      chalk.white('🎯 **Job Fit Analysis**\n') + chalk.gray('  • "analyze this job: [paste URL]"\n') +
      chalk.white('📱 **Contact Generation**\n') + chalk.gray('  • "help me contact via email"\n') +
      chalk.white('🌐 **Website & Resume**\n') + chalk.gray('  • "latest blog posts"\n') +
      chalk.cyan.bold('💡 Example Commands:\n') + chalk.white('  • "Show me Akshit\'s latest GitHub projects"\n'),
      { padding: 1, margin: 1, borderStyle: 'round', borderColor: 'green', backgroundColor: '#1a1a1a' }
    ));
  }
}

wss.on('connection', (ws) => {
  console.log('[chatbot-server] WebSocket client connected');

  const bot = new AkshitChatbot((msg) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(msg);
  });

  bot.start();

  ws.on('message', (msg) => {
    const text = msg.toString();
    bot.handleInput(text);
  });
});

process.on('SIGINT', () => {
  console.log('[chatbot-server] SIGINT received, shutting down...');
  server.close(() => process.exit(0));
});

module.exports = { app, server };