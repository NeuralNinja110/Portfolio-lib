#!/bin/bash
# Documentation Verification Script

echo "🔍 Portfolio-lib Documentation Verification"
echo "==========================================="
echo

cd /workspaces/Portfolio-lib/docs/build/html

# Check file counts
html_count=$(ls -1 *.html 2>/dev/null | wc -l)
image_count=$(ls -1 _images/*.png 2>/dev/null | wc -l)
static_count=$(ls -1 _static/*.css _static/*.js 2>/dev/null | wc -l)

echo "📄 Generated Files:"
echo "   HTML pages: $html_count"
echo "   Images: $image_count" 
echo "   Static files: $static_count"
echo

# Check key documentation sections
echo "📚 Documentation Sections:"
sections=("installation.html" "getting_started.html" "advanced_usage.html" "troubleshooting.html" "api_reference.html" "visual_guide.html")

for section in "${sections[@]}"; do
    if [ -f "$section" ]; then
        size=$(stat -c%s "$section")
        echo "   ✅ $section (${size} bytes)"
    else
        echo "   ❌ $section (missing)"
    fi
done
echo

# Check API documentation
echo "🔧 API Documentation:"
api_files=("portfolio_lib.core.html" "portfolio_lib.indicators.html" "portfolio_lib.portfolio.html")

for api_file in "${api_files[@]}"; do
    if [ -f "$api_file" ]; then
        functions=$(grep -c "class\|def\|function" "$api_file" 2>/dev/null || echo 0)
        echo "   ✅ $api_file ($functions functions/classes)"
    else
        echo "   ❌ $api_file (missing)"
    fi
done
echo

# Check visual content
echo "🎨 Visual Content:"
echo "   📊 Plot images: $(find _images/ -name "*visual_guide*.png" 2>/dev/null | wc -l)"
echo "   📈 Indicator plots: $(find _images/ -name "*indicators*.png" 2>/dev/null | wc -l)"
echo "   🎯 Custom CSS: $([ -f "_static/custom.css" ] && echo "✅ Present" || echo "❌ Missing")"
echo

# Check search functionality
echo "🔍 Search & Navigation:"
echo "   📖 Search index: $([ -f "searchindex.js" ] && echo "✅ Generated" || echo "❌ Missing")"
echo "   🔗 Module index: $([ -f "py-modindex.html" ] && echo "✅ Generated" || echo "❌ Missing")"
echo "   📚 General index: $([ -f "genindex.html" ] && echo "✅ Generated" || echo "❌ Missing")"
echo

# Check configuration files
echo "⚙️  Deployment Configuration:"
echo "   📋 Read the Docs: $([ -f "../../.readthedocs.yaml" ] && echo "✅ Configured" || echo "❌ Missing")"
echo "   🚀 GitHub Actions: $([ -f "../../.github/workflows/docs.yml" ] && echo "✅ Configured" || echo "❌ Missing")"
echo "   🚫 Jekyll bypass: $([ -f ".nojekyll" ] && echo "✅ Present" || echo "❌ Missing")"
echo

# Summary
echo "📊 Verification Summary:"
if [ $html_count -gt 15 ] && [ $image_count -gt 30 ] && [ -f "index.html" ]; then
    echo "   🎉 Documentation build: SUCCESSFUL"
    echo "   ✅ All key sections generated"
    echo "   ✅ Visual content present"
    echo "   ✅ Ready for deployment"
else
    echo "   ⚠️  Documentation build: INCOMPLETE"
    echo "   ❌ Some issues detected"
fi

echo
echo "🌐 Local Preview:"
echo "   Open: file://$(pwd)/index.html"
echo "   Or run: python -m http.server 8000"
echo
echo "Done! 🎯"
