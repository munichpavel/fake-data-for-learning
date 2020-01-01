# Git add and commit with message docs changes
commit_message=$1

if [[ -n "$commit_message" ]]; then
    echo "Change to local builddir" $LOCAL_BUILDDIR
    cd $LOCAL_BUILDDIR

    echo "Git-cleaning built docs"
    git pull origin master
    git clean -df

    echo "Adding modifications"
    git add .
    echo "Committing with message: $commit_message"
    git commit -m "$commit_message"
    git push origin master
    cd -

else
    echo "Aborting local git changes: argument error"
fi

