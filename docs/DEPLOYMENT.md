# Documentation Deployment Guide

This guide explains how to deploy the portfolio-lib documentation to Read the Docs and GitHub Pages.

## Quick Start

The documentation is configured for both Read the Docs and GitHub Pages deployment. Choose your preferred hosting method:

### GitHub Pages (Recommended)

1. **Enable GitHub Pages in your repository:**
   - Go to your repository settings
   - Navigate to "Pages" section
   - Source: GitHub Actions
   - The workflow in `.github/workflows/docs.yml` will automatically build and deploy

2. **Automatic deployment triggers:**
   - Push to `main` or `master` branch
   - Pull requests (for testing)

### Read the Docs

1. **Connect your repository:**
   - Sign up at [readthedocs.org](https://readthedocs.org)
   - Import your GitHub repository
   - Configuration is automatically detected from `.readthedocs.yaml`

2. **Build configuration:**
   - Python version: 3.11
   - Sphinx configuration: `docs/source/conf.py`
   - Requirements: `docs/requirements.txt`

## Configuration Files

### Read the Docs (.readthedocs.yaml)
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
   configuration: docs/source/conf.py
python:
   install:
   - requirements: docs/requirements.txt
   - requirements: portfolio-lib-package/requirements.txt
   - method: pip
     path: portfolio-lib-package
```

### GitHub Actions (.github/workflows/docs.yml)
- Builds documentation on push/PR
- Deploys to GitHub Pages automatically
- Uses Python 3.11 and latest dependencies

### Sphinx Configuration (docs/source/conf.py)
- Read the Docs theme
- GitHub integration
- Custom CSS styling
- Plot directive for matplotlib
- MathJax for formulas

## Manual Build Instructions

### Local Development
```bash
cd docs
make clean
make html
# Open docs/build/html/index.html in browser
```

### Production Build
```bash
# Install dependencies
pip install -r docs/requirements.txt
pip install -r portfolio-lib-package/requirements.txt
pip install -e portfolio-lib-package/

# Build documentation
cd docs
make clean
make html

# Serve locally (optional)
python -m http.server 8000 -d build/html
```

## Customization Options

### Theme Customization
- Edit `docs/source/_static/custom.css` for styling
- Modify `html_theme_options` in `conf.py`
- Add logo/favicon by updating paths in `conf.py`

### Content Updates
- Add new documentation files in `docs/source/`
- Update `index.rst` toctree to include new files
- Rebuild documentation

### GitHub Integration
Update the following in `conf.py` for your repository:
```python
html_context = {
    "github_user": "NeuralNinja110",
    "github_repo": "Portfolio-lib",
    "github_version": "main",
}
```

## Troubleshooting

### Build Errors
- Check Python dependencies in `requirements.txt`
- Verify package installation paths
- Review Sphinx warnings in build output

### Deployment Issues
- Ensure repository is public for GitHub Pages
- Check GitHub Actions logs for build failures
- Verify Read the Docs build logs

### Missing Content
- Run `sphinx-apidoc` to regenerate API documentation
- Check file paths in toctree directives
- Verify plot directive examples execute correctly

## Advanced Features

### Internationalization
- Add language codes to `conf.py`
- Create translated documentation in `docs/source/locale/`

### PDF Generation
- Add `formats: [pdf]` to `.readthedocs.yaml`
- Requires LaTeX dependencies

### Search Enhancement
- Enable search analytics in theme options
- Customize search behavior in `conf.py`

## Maintenance

### Regular Updates
- Update dependencies in `requirements.txt`
- Rebuild documentation after code changes
- Monitor build logs for warnings/errors

### Performance Optimization
- Optimize image sizes in `_static/` and `_images/`
- Use plot directive caching for matplotlib figures
- Consider CDN for static assets

## Support

For deployment issues:
1. Check GitHub Actions/Read the Docs build logs
2. Review Sphinx documentation for configuration options
3. Test builds locally before deployment
4. Submit issues to the portfolio-lib repository
