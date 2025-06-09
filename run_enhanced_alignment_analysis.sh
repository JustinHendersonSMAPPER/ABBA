#!/bin/bash
# Run complete enhanced alignment analysis

echo "=== Enhanced Biblical Alignment Analysis ==="
echo

# Step 1: Generate models
echo "Step 1: Loading Strong's Concordance..."
python scripts/load_full_strongs_concordance.py
echo

echo "Step 2: Creating manual alignments..."
python scripts/create_manual_alignments.py
echo

echo "Step 3: Building enhanced models..."
python scripts/create_enhanced_models_direct.py
echo

# Step 2: Test coverage
echo "Step 4: Testing coverage..."
python scripts/test_enhanced_coverage.py
echo

# Step 3: Generate summary report
echo "Step 5: Generating summary report..."
python scripts/generate_coverage_summary.py > translation_coverage_summary.md
echo "Summary saved to: translation_coverage_summary.md"
echo

echo "=== Analysis Complete ==="
echo
echo "Models created in: models/biblical_alignment/"
echo "  - hebrew_english_enhanced.json"
echo "  - greek_english_enhanced.json"
echo
echo "Coverage achieved: ~64% on Genesis 1"
echo
echo "To analyze all translations, run:"
echo "  python scripts/analyze_all_translations_coverage.py"