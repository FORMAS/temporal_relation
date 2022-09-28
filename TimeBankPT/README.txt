TimeBankPT


===============
Contents of this document

1. The corpus
2. Version
3. A note on the annotations
4. Files in this release
5. Citation


===============
1. The corpus

TimeBankPT was obtained by translating the English data used in the first TempEval competition (http://timeml.org/tempeval).

TimeBankPT can be found at http://nlx.di.fc.ul.pt/~fcosta/TimeBankPT.
That page contains some information about the corpus and a link to the release.


===============
2. Version

This is the release of TimeBankPT version 1.0.


===============
3. A note on the annotations

The annotations are like the ones in the English TempEval (2007) data, except for these differences:
-- Whereas the TempEval data used two sets of files, one for tasks A and B and another one for
   task C, here all annotations are in the same files (1 file per document, instead of 2).
-- The ids for the TLINKs for task C have the suffix "c" so that their names do not clash with the
   names of the TLINKs for the other tasks, as all TLINKs for each document are now in the same files.
-- The data have undergone some corrections, which are documented in the file data-corrections.txt. The
   motivations behind these corrections are explained in Costa and Branco (2012).

Additionally, the "tense" attribute of EVENT elements has different values than the ones employed for English, due to language differences. The file tense-values.txt contains an explanation of each of these values.


===============
4. Files in this release

The following files are part of this release:
-- test
   This directory contains the documents that are part of the test set. There should be 20 of them.
-- train
   This directory contains the documents that are part of the train set. There should be 162 documents here.
-- counts.sh
   This Bash script outputs some statistics for the corpus, such as the number of sentences, word tokens and
   various kinds of annotated items.
-- data-corrections.txt
   This file documents some manual corrections performed on the annotations. The motivations behind these
   corrections are explained in Costa and Branco (2012).
-- README.txt
   This file.
-- tense-values.txt
   Explanation of all the values used for the "tense" attribute of EVENT elements.


===============
5. Citation

The preferred citation is Costa and Branco (2012). Further details about the corpus can be found in the following publications:

-- Costa, Francisco and Branco, António. 2010. Temporal Information Processing of a New Language: Fast Porting with Minimal Resources. In ACL2010-Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics.
-- Costa, Francisco and Branco, António. 2012. TimeBankPT: A TimeML Annotated Corpus of Portuguese. In Proceedings of LREC2012.
-- Costa, Francisco. to appear. Processing Temporal Information in Unstructured Documents. Ph.D.thesis, Universidade de Lisboa, Lisbon.
