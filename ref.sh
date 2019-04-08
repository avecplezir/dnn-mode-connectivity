git filter-branch -f --tree-filter  'rm -rf flows_checkpoints/IAFjn/checkpoint-50.pt' HEAD

git filter-branch -f --tree-filter  'rm -rf flows_checkpoints/RNVP/checkpoint-100.pt' HEAD

git filter-branch -f --tree-filter  'rm -rf flows_checkpoints/RNVPjn/checkpoint-150.pt' HEAD

git filter-branch -f --tree-filter  'rm -rf flows_checkpoints/RNVPjn/checkpoint-50.pt' HEAD
