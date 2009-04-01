#!/bin/sh

EXECUTABLE=$1

BASENAME=`basename $EXECUTABLE`

# The top-level directory name
TOPDIR=..

# Set the profile directory name
PROFILEDIR=$TOPDIR/profile

# Make the profile directory if it doesn't already exist
if test ! -d $PROFILEDIR; then mkdir $PROFILEDIR; fi

# Get the current working revision
REVISION=`svn info -r HEAD | grep 'Revision' | sed 's/Revision: //'`

LOGFILENAME=$PROFILEDIR/$BASENAME-w$REVISION.log
if test -f $LOGFILENAME; then rm -f $LOGFILENAME; fi

echo -n $BASENAME"... ";
(echo $HOSTNAME; date; echo; bash $EXECUTABLE) 1>  $LOGFILENAME
echo "done."

# Test if the current working revision is untouched
if svn status ../ | grep '^M' > /dev/null
then
    # Revision has been changed since last checked in
    # Write output to file with name $EXECUTABLE-w$REVISION.NUMBER.log
    NUMBER=1
    COPYLOGFILENAME=$PROFILEDIR/$BASENAME-w$REVISION.0$NUMBER.log
    # Test for previous log file
    while test -f $COPYLOGFILENAME
    do
        NUMBER=$[$NUMBER+1]
        if test "$NUMBER" -lt 10; then
            # Pad number with leading zero to preserve order of filenames
            COPYLOGFILENAME=$PROFILEDIR/$BASENAME-w$REVISION.0$NUMBER.log
        else
            COPYLOGFILENAME=$PROFILEDIR/$BASENAME-w$REVISION.$NUMBER.log
        fi
    done
    cp $LOGFILENAME $COPYLOGFILENAME
    echo $COPYLOGFILENAME
else
    # Write output to file with name $EXECUTABLE-r$REVISION.log
    MOVELOGFILENAME=$PROFILEDIR/$BASENAME-r$REVISION.log
    mv -f$LOGFILENAME $MOVELOGFILENAME
    echo $MOVELOGFILENAME
fi

echo $LOGFILENAME


