#!/bin/bash

cat reference_trees/aa_rokasA1.reference  aa_rokasA1_48@2*/*.bestTree    > reference_trees/aa_rokasA1_48@2.allTrees
cat reference_trees/aa_rokasA4.reference  aa_rokasA4_40@100*/*.bestTree  > reference_trees/aa_rokasA4_40@100.allTrees
cat reference_trees/aa_rokasA8.reference  aa_rokasA8_48@2*/*.bestTree    > reference_trees/aa_rokasA8_48@2.allTrees
cat reference_trees/dna_rokasD1.reference dna_rokasD1_48@10*/*.bestTree  > reference_trees/dna_rokasD1_48@10.allTrees
cat reference_trees/dna_rokasD4.reference dna_rokasD4_48@4*/*.bestTree   > reference_trees/dna_rokasD4_48@4.allTrees
cat reference_trees/dna_rokasD7.reference dna_rokasD7_40@100*/*.bestTree > reference_trees/dna_rokasD7_40@100.allTrees
cat reference_trees/dna_PeteD8.reference  dna_PeteD8_48@5*/*.bestTree    > reference_trees/dna_PeteD8_48@5.allTrees

for f in reference_trees/*.allTrees; do
    numberOfTopologies=$(./raxml-reference --rfdist "$f" --redo | grep "Number of unique topologies in this tree set: ")
    echo "$f: $numberOfTopologies"
done
