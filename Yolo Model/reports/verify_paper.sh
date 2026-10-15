#!/bin/bash

echo "=========================================="
echo "IEEE LaTeX Paper Verification Report"
echo "=========================================="
echo ""

echo "1. File Structure:"
echo "   Main document: $(ls -lh main.tex | awk '{print $9, $5}')"
echo "   Compiled PDF:  $(ls -lh main.pdf | awk '{print $9, $5}')"
echo "   Bibliography:  $(ls -lh references.bib | awk '{print $9, $5}')"
echo ""

echo "2. Section Files:"
for f in sections/*.tex; do
    lines=$(wc -l < "$f")
    echo "   $(basename $f): $lines lines"
done
echo ""

echo "3. Figures:"
for f in fig*.png; do
    if [ -f "$f" ]; then
        echo "   $f: $(file $f | cut -d: -f2)"
    fi
done
echo ""

echo "4. PDF Information:"
echo "   Pages: $(grep -o "Output written on main.pdf ([0-9]* pages" main.log | grep -o "[0-9]*" | head -1)"
echo "   Size: $(ls -lh main.pdf | awk '{print $5}')"
echo ""

echo "5. Citation Statistics:"
echo "   Total \\cite commands: $(grep -r "\\\\cite{" sections/ | wc -l)"
echo "   Unique citations: $(grep -r "\\\\cite{" sections/ | grep -o "\\\\cite{[^}]*}" | sort -u | wc -l)"
echo "   Bibliography entries: $(grep -c "^@" references.bib)"
echo ""

echo "6. Table Count:"
echo "   Tables in document: $(grep -r "\\\\begin{table}" sections/ | wc -l)"
echo ""

echo "7. Equation Count:"
echo "   Equations in document: $(grep -r "\\\\begin{equation}" sections/ | wc -l)"
echo ""

echo "8. Compilation Status:"
if [ -f main.pdf ]; then
    echo "   ✓ PDF successfully generated"
else
    echo "   ✗ PDF generation failed"
fi

if [ -f main.bbl ]; then
    echo "   ✓ Bibliography processed"
else
    echo "   ✗ Bibliography not processed"
fi
echo ""

echo "9. Warnings/Errors:"
error_count=$(grep -c "^!" main.log 2>/dev/null || echo "0")
warning_count=$(grep -c "Warning:" main.log 2>/dev/null || echo "0")
echo "   Errors: $error_count"
echo "   Warnings: $warning_count"
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
