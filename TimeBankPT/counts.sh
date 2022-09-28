#!/bin/bash
#
# Bash script that counts the number of sentences, words
# annotated events, timexes and tlinks in TimeBankPT
#

THIS_DIR=$PWD
TRAIN_DIR="train"
TEST_DIR="test"

# COUNT SENTENCES
SENTENCE_REGEX="<s>"

sentence_count_train=`cd $TRAIN_DIR && grep -o $SENTENCE_REGEX * | wc -l && cd $THIS_DIR `
sentence_count_test=`cd $TEST_DIR && grep -o $SENTENCE_REGEX * | wc -l && cd $THIS_DIR `

# COUNT WORD TOKENS
# according to whitespace
# ignoring TimeML tags

word_count_train=`cd $TRAIN_DIR && cat * | sed 's/<[^>]*>//g' | wc -w && cd $THIS_DIR `
word_count_test=`cd $TEST_DIR && cat * | sed 's/<[^>]*>//g' | wc -w && cd $THIS_DIR `

# COUNT EVENTS, TIMEXES AND TLINKS
EVENT_REGEX="<EVENT"
TIMEX_REGEX="<TIMEX"
TLINK_REGEX="<TLINK[^>]*>"
TASK_A_REGEX="task=\"A\""
TASK_B_REGEX="task=\"B\""
TASK_C_REGEX="task=\"C\""

event_count_train=`cd $TRAIN_DIR && cat * | grep -o $EVENT_REGEX | wc -l && cd $THIS_DIR`
event_count_test=`cd $TEST_DIR && cat * | grep -o $EVENT_REGEX | wc -l && cd $THIS_DIR`

timex_count_train=`cd $TRAIN_DIR && cat * | grep -o $TIMEX_REGEX | wc -l && cd $THIS_DIR`
timex_count_test=`cd $TEST_DIR && cat * | grep -o $TIMEX_REGEX | wc -l && cd $THIS_DIR`

tlink_count_train=`cd $TRAIN_DIR && cat * | grep -o $TLINK_REGEX | wc -l && cd $THIS_DIR`
tlink_count_test=`cd $TEST_DIR && cat * | grep -o $TLINK_REGEX | wc -l && cd $THIS_DIR`

task_a_count_train=`cd $TRAIN_DIR && cat * | grep -o $TLINK_REGEX | grep $TASK_A_REGEX | wc -l && cd $THIS_DIR`
task_b_count_train=`cd $TRAIN_DIR && cat * | grep -o $TLINK_REGEX | grep $TASK_B_REGEX | wc -l && cd $THIS_DIR`
task_c_count_train=`cd $TRAIN_DIR && cat * | grep -o $TLINK_REGEX | grep $TASK_C_REGEX | wc -l && cd $THIS_DIR`

task_a_count_test=`cd $TEST_DIR && cat * | grep -o $TLINK_REGEX | grep $TASK_A_REGEX | wc -l && cd $THIS_DIR`
task_b_count_test=`cd $TEST_DIR && cat * | grep -o $TLINK_REGEX | grep $TASK_B_REGEX | wc -l && cd $THIS_DIR`
task_c_count_test=`cd $TEST_DIR && cat * | grep -o $TLINK_REGEX | grep $TASK_C_REGEX | wc -l && cd $THIS_DIR`


# PRINT THE COUNTS
echo "Number of Sentences"
echo "  Train: " $sentence_count_train
echo "  Test: " $sentence_count_test
echo "Number of Words"
echo "  Train: " $word_count_train
echo "  Test: " $word_count_test
echo "Number of Annotated Events"
echo "  Train: " $event_count_train
echo "  Test: " $event_count_test
echo "Number of Annotated Temporal Expressions"
echo "  Train: " $timex_count_train
echo "  Test: " $timex_count_test
echo "Number of Annotated Temporal Relations"
echo "  Train: " $tlink_count_train
echo "  Test: " $tlink_count_test
echo "Number of Annotated Task A Temporal Relations"
echo "  Train: " $task_a_count_train
echo "  Test: " $task_a_count_test
echo "Number of Annotated Task B Temporal Relations"
echo "  Train: " $task_b_count_train
echo "  Test: " $task_b_count_test
echo "Number of Annotated Task C Temporal Relations"
echo "  Train: " $task_c_count_train
echo "  Test: " $task_c_count_test

