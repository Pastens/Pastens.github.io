#!/bin/bash
# Sync Obsidian Knowledge vault → Quartz content/
# Handles nested subdirectories recursively.
# Run after writing new notes in Obsidian, then commit & push.

set -e

VAULT="/mnt/c/Users/admin/Documents/Obsidian Vault/Knowledge"
CONTENT=~/Pastens.github.io/content

echo "🔄 Syncing Obsidian vault → Quartz content..."

# Recursively copy markdown files preserving relative structure
cd "$VAULT"
find . -name "*.md" -type f | while IFS= read -r file; do
  target="$CONTENT/$file"
  mkdir -p "$(dirname "$target")"
  cp "$VAULT/$file" "$target"
done

total=$(find "$CONTENT" -name "*.md" -type f | wc -l)
echo "✅ Sync complete! ($total markdown files in content/)"
echo ""
echo "Next steps:"
echo "  cd $CONTENT/.."
echo "  git add content/ && git commit -m 'sync: update content from Obsidian vault'"
echo "  git push"
