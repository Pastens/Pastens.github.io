#!/bin/bash
# Sync Obsidian Knowledge vault → Quartz content/
# Run this after writing new notes in Obsidian, then commit & push

set -e

VAULT="/mnt/c/Users/admin/Documents/Obsidian Vault/Knowledge"
CONTENT=~/Pastens.github.io/content

echo "🔄 Syncing Obsidian vault → Quartz content..."

# Sync each category directory
for dir in "$VAULT"/*/; do
  category=$(basename "$dir")
  target="$CONTENT/$category"
  mkdir -p "$target"
  cp "$dir"*.md "$target/" 2>/dev/null && echo "  ✓ $category ($(ls "$dir"*.md 2>/dev/null | wc -l) files)" || echo "  - $category (empty)"
done

# Sync top-level markdown files in Knowledge
cp "$VAULT"/*.md "$CONTENT/" 2>/dev/null

echo "✅ Sync complete!"
echo ""
echo "Next steps:"
echo "  cd ~/Pastens.github.io"
echo "  git add content/ && git commit -m 'sync: update content from Obsidian vault'"
echo "  git push"
