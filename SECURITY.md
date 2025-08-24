# Security Policy

## Reporting Security Vulnerabilities

We take the security of DocuChat seriously. If you believe you have found a security vulnerability, we encourage you to report it to us responsibly.

### How to Report

1. **GitHub Issues**: For non-sensitive security issues, please create a GitHub issue with the label "security"
2. **Private Reports**: For sensitive vulnerabilities, please create a private GitHub issue or contact the maintainers directly

### What to Include

When reporting a security issue, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes or mitigations
- Your contact information (optional)

### Response Timeline

- **Initial Response**: We aim to acknowledge security reports within 48 hours
- **Status Updates**: We will provide regular updates on the progress of addressing the issue
- **Resolution**: We strive to resolve security issues within 30 days of initial report

### Security Best Practices for Users

When using DocuChat, we recommend:

1. **Document Security**: Be cautious when processing sensitive documents
2. **API Keys**: Keep your LLM API keys secure and never commit them to version control
3. **Network Security**: When deploying the web interface, ensure proper network security measures
4. **Updates**: Keep DocuChat updated to the latest version to receive security patches
5. **Environment Variables**: Use environment variables for sensitive configuration data

### Supported Versions

We provide security updates for:

- The latest stable release
- The current development branch

### Disclosure Policy

- We request that you give us reasonable time to address the issue before public disclosure
- We will acknowledge your contribution to improving DocuChat's security
- We may publish a security advisory after the issue is resolved

## Security Features

DocuChat includes the following security considerations:

- Local document processing by default (documents stay on your system)
- Configurable API endpoints for LLM services
- No telemetry or data collection by default
- Open source codebase for transparency

Thank you for helping keep DocuChat secure!