#!/bin/bash
# Sync Obsidian Knowledge vault → Quartz content/
# Preserves the full vault-relative path (Knowledge/ prefix) so
# [[wikilinks]] resolve correctly in Quartz.
# Run after writing new notes in Obsidian, then commit & push.

set -e

VAULT="/mnt/c/Users/admin/Documents/Obsidian Vault"
CONTENT=~/Pastens.github.io/content

cd "$VAULT"

echo "🔄 Syncing Obsidian vault → Quartz content..."

find Knowledge -name "*.md" -type f | while IFS= read -r file; do
  target="$CONTENT/$file"
  mkdir -p "$(dirname "$target")"
  cp "$VAULT/$file" "$target"
done

total=$(find "$CONTENT/Knowledge" -name "*.md" -type f | wc -l)
echo "✅ Sync complete! ($total markdown files in content/Knowledge/)"
echo ""
echo "Next steps:"
echo "  cd $CONTENT/.."
echo "  git add content/ && git commit -m 'sync: update content from Obsidian vault'"
echo "  git push"
