# Security Guidelines

## 🔐 API Key Management

### For Hugging Face API
When using `tilde_api_demo.py`, you'll be prompted for an optional API token. **NEVER** commit API keys to the repository.

**Safe practices:**
```bash
# Set as environment variable
export HF_TOKEN="your_token_here"

# Use in Python
import os
api_token = os.getenv('HF_TOKEN')
```

### For Cloud Deployments
- Use platform-specific secret management (GitHub Secrets, HF Spaces secrets, etc.)
- Never hardcode credentials in source code
- Use environment variables for sensitive data

## 🛡️ Privacy Considerations

### Data Processing
- All model interactions happen on remote servers (HF, Colab, etc.)
- Your conversations are processed by third-party services
- Consider data sensitivity when using cloud platforms

### Local vs Cloud
- **Local execution**: Data stays on your machine
- **Cloud execution**: Data is sent to cloud providers
- **API usage**: Data goes through Hugging Face infrastructure

## 🚨 What's Safe in This Repository

✅ **Safe to make public:**
- Model loading code without credentials
- Configuration files without secrets
- Documentation and examples
- Open source dependencies

❌ **Never commit:**
- API keys or tokens
- Personal information
- Private model weights
- Credentials of any kind

## 🔒 Recommended Practices

1. **Use environment variables** for all secrets
2. **Review commits** before pushing
3. **Use .gitignore** to exclude sensitive files
4. **Rotate tokens** regularly
5. **Use minimal permissions** for API keys

## 📋 Security Checklist

Before making repository public:
- [ ] No hardcoded API keys
- [ ] No personal information
- [ ] Placeholder usernames replaced
- [ ] .gitignore configured
- [ ] Environment variables documented

## 🆘 If You Accidentally Commit Secrets

1. **Immediately rotate** the compromised credentials
2. **Remove from Git history** using `git filter-branch` or BFG Repo-Cleaner
3. **Force push** the cleaned history
4. **Update any systems** using the old credentials

## 📞 Reporting Security Issues

If you find security vulnerabilities, please:
1. **Do not** open a public issue
2. **Contact** the repository maintainer privately
3. **Provide** detailed information about the vulnerability
4. **Allow** reasonable time for fixing before public disclosure
