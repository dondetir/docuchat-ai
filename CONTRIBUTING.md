# Contributing to DocuChat

Thank you for your interest in contributing to DocuChat! We welcome contributions from the community and appreciate your effort to make this project better.

## 🚀 Quick Start for Contributors

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/dondetir/docuchat-ai.git
cd docuchat-ai
```

2. **Set up Development Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install pytest black flake8
```

3. **Test Your Setup**
```bash
# Run tests to ensure everything works
python -m pytest tests/

# Start the web interface
python web/run_web_app.py
```

## 🛠️ Development Guidelines

### Code Standards

- **Python Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type hints for all function parameters and return values
- **Documentation**: Add docstrings to all public functions and classes
- **Error Handling**: Include proper exception handling and meaningful error messages

### Testing

- Write tests for new features and bug fixes
- Ensure all existing tests pass before submitting PR
- Include both unit tests and integration tests where appropriate
- Test with different document types and edge cases

### Commit Messages

Follow conventional commit format:
```
type(scope): brief description

Detailed explanation if needed

- Include bullet points for multiple changes
- Reference issue numbers when applicable
```

Examples:
- `feat(web): add dark mode toggle to interface`
- `fix(chunker): resolve memory leak in large document processing`
- `docs: update installation instructions for Docker`

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment Details**
   - Operating system and version
   - Python version
   - DocuChat version/commit hash
   - Ollama version

2. **Reproduction Steps**
   - Clear steps to reproduce the issue
   - Sample documents or data (if applicable and not sensitive)
   - Expected vs actual behavior

3. **Error Messages**
   - Full error messages and stack traces
   - Relevant log outputs (use `--verbose` flag)

### Feature Requests

For feature requests, please:

1. Check if the feature already exists or is planned
2. Describe the use case and problem it solves
3. Suggest potential implementation approaches
4. Consider backward compatibility

## 🔧 Types of Contributions

### Code Contributions

- **Bug fixes**: Fix existing issues
- **New features**: Implement new functionality
- **Performance improvements**: Optimize existing code
- **Refactoring**: Improve code structure and maintainability

### Documentation

- **API documentation**: Improve docstrings and code comments
- **User guides**: Enhance README and usage documentation
- **Tutorials**: Create examples and how-to guides
- **Translation**: Help translate documentation

### Testing

- **Unit tests**: Add tests for individual components
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark and stress testing
- **Cross-platform testing**: Test on different operating systems

## 📝 Pull Request Process

### Before Submitting

1. **Create an Issue**: For major changes, create an issue first to discuss
2. **Branch Naming**: Use descriptive branch names (`feature/add-pdf-ocr`, `fix/memory-leak`)
3. **Code Quality**: Ensure code follows our standards
4. **Tests**: Add/update tests for your changes
5. **Documentation**: Update documentation if needed

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Tested with different document types
- [ ] Tested on different platforms (if applicable)

## Checklist
- [ ] Code follows the project's coding standards
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No unnecessary console.log statements or debug code
```

### Review Process

1. **Automated Checks**: Ensure CI/CD passes
2. **Code Review**: At least one maintainer will review
3. **Testing**: Changes will be tested in various environments
4. **Merge**: Once approved, changes will be merged

## 🏗️ Project Structure

Understanding the codebase:

```
docuchat-ai/
├── src/                    # Core application modules
│   ├── document_loader.py  # Document parsing and loading
│   ├── chunker.py          # Text chunking logic
│   ├── embeddings.py       # Embedding generation
│   ├── vector_db.py        # Vector database operations
│   ├── llm_client.py       # LLM integration
│   └── rag_pipeline.py     # Main RAG orchestration
├── web/                    # Web interface
│   ├── web_app.py          # Gradio web application
│   └── run_web_app.py      # Web app launcher
├── tests/                  # Test suite
├── deployment/             # Docker and deployment configs
├── docs/                   # Documentation
└── docuchat.py            # CLI entry point
```

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- **Performance optimization**: Faster document processing
- **Memory efficiency**: Better handling of large document sets
- **Error handling**: More robust error recovery
- **Cross-platform compatibility**: Windows/Mac/Linux testing

### Medium Priority
- **New document formats**: Support for more file types
- **UI improvements**: Enhanced web interface features
- **Docker optimization**: Smaller, faster containers
- **Monitoring**: Health checks and metrics

### Community Requests
- **Mobile support**: Responsive design improvements
- **Batch operations**: Process multiple document sets
- **API endpoints**: RESTful API for programmatic access
- **Plugins**: Extension system for custom processors

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome contributors of all skill levels
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone is learning

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and discussions
- **Discussions**: For general questions and community chat
- **Email**: For security issues or sensitive topics

## 🏆 Recognition

### Contributors

All contributors will be recognized in:
- GitHub contributors list
- Release notes (for significant contributions)
- Documentation acknowledgments

### Becoming a Maintainer

Active contributors may be invited to become maintainers based on:
- Consistent quality contributions
- Understanding of project architecture
- Community involvement and helpfulness
- Long-term commitment to the project

## 📚 Resources

### Learning Resources

- **RAG Systems**: [Retrieval-Augmented Generation Guide](https://huggingface.co/docs/transformers/model_doc/rag)
- **Vector Databases**: [ChromaDB Documentation](https://docs.trychroma.com/)
- **LLM Integration**: [Ollama Documentation](https://github.com/jmorganca/ollama)
- **Python Best Practices**: [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

### Development Tools

- **Code Formatting**: Use `black` for Python formatting
- **Linting**: Use `flake8` for code quality checks
- **Testing**: Use `pytest` for running tests
- **Documentation**: Use docstrings following Google style

## 🙋 Getting Help

### For Contributors

- **Documentation**: Check existing docs first
- **Issues**: Search existing issues for similar problems
- **Discussions**: Ask questions in GitHub Discussions
- **Code Review**: Request reviews from maintainers

### For Maintainers

- **New Contributor Guide**: Help onboard new contributors
- **Code Review**: Provide constructive feedback
- **Issue Triage**: Help categorize and prioritize issues
- **Release Management**: Assist with version releases

---

## 🎉 Thank You!

Your contributions make DocuChat better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, your efforts are appreciated and make a real difference in the community.

**Happy coding!** 🚀