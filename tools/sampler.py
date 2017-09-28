#!/usr/bin/env python

import argparse
import os
import numpy as np
import codecs
from pprint import pprint
from collections import deque
from nltk.tree import Tree
from nltk.grammar import Production, ProbabilisticProduction, Nonterminal, PCFG, is_terminal
import nltk.data
from .grammar import *
import pdb

#############################################################
# Sampler 
#############################################################

class Sampler(object):
  """
  Sample from a given grammar
  """
  
  def __init__(self, grammar):
    """
    Initialize the tools with a grammar to sample from

    :param grammar: Path to the grammar
    :type grammar: string       
    """
    self.load_grammar(grammar)

  def load_grammar(self, grammar):
    """
    Load a grammar. Currently PCFG is supported only.
    
    :param grammar: Path to the grammar
    :type grammar: string    
    """
    # self.grammar = nltk.data.load(grammar, 'pcfg')
    self.grammar = SCFG()
    self.grammar.load_from_grammar_file(grammar)
    
  def sample_production(self, lhs):
    """
    Sample a production from the grammar starting with lhs
    
    :param lhs: The left-hand side symbol
    :type lhs: Nonterminal
    """
  
    # the productions with lhs s and their probabilities
    choices = list(self.grammar.productions_filtered(lhs))    
    probabilities = [prod.prob() for prod in choices] 
    # make a random choice
    choice = np.random.choice(len(choices), size=1, p=probabilities)

    # TODO make verbose
    # print("choice: %d (%s) from probabilities: %s" % (choice, choices[choice],
      # " ".join([str(p) for p in probabilities])))
    
    return choices[choice]
    
  def derivation_to_tree(self, d, node, node_id):
    """
    Convert a derivation map to an NLTK tree
    """
    irhs_with_id, orhs = d[(node, node_id)]
    
    children = []

    for x in irhs_with_id:
      
      child, child_id = x
      # print("process_der:", child, child_id)
      
      if is_terminal(child):
        children.append(child)
      else:
        children.append(self.derivation_to_tree(d, child, child_id))
    
    t = Tree(node, children)
    return t
  

  def derivation_to_tree_orhs(self, d, node, node_id):
    """
    Convert a derivation map to an NLTK tree - orhs version
    """
    irhs_with_id, orhs = d[(node, node_id)]
    

    children = []

    for child_rhs in orhs:

      if isinstance(child_rhs,int):
        child, child_id = list(filter(lambda x : not is_terminal(x[0]),  irhs_with_id))[child_rhs-1] 
        children.append(self.derivation_to_tree_orhs(d, child, child_id))
      elif is_terminal(child_rhs):
        children.append(child_rhs)
    
    t = Tree(node, children)
    return t

  def sample(self, max_seq_len):
    """
    Sample a derivation from the grammar
    """

    q = deque() # queue
    d = {}      # derivation
    i = 0       # lhs index
    t_count = 0 # number of generated terminals (so far)
  
    q.append((Nonterminal(args.start), i)) # append start symbol
    i += 1

    while len(q) > 0:

      # stop if this sequence is going to be longer than the
      # requested sequence length (we discard it anyway)
      if t_count > max_seq_len:
        return None

      lhs, lhs_id = q.popleft() # Nonterminal, ID
    
      # print("processing: %s (%d)" % (lhs, lhs_id))
    
      if not is_terminal(lhs):

        # choose a production with u as lhs
        r = self.sample_production(lhs)

        # count number of terminals
        for item in r.irhs():
          if is_terminal(item):
            t_count += 1

        # create a production-rhs where the symbols all get a unique ID
        # so we can reconstruct the derivation later
        irhs = list(zip(r.irhs(), range(i, i + len(r))))
        i += len(r)
        
        # add the non-terminals to the queue
        q.extend(irhs)
        
        # save to the derivation
        d[(lhs, lhs_id)] = tuple([irhs, r.orhs()])
    
    return d  
    #return self.derivation_to_tree_orhs(d, Nonterminal(args.start), 0)     

########################################################################
# Main
########################################################################

def handle_args():  
  parser = argparse.ArgumentParser(description='Sample from a grammar.')
  parser.add_argument('grammar', metavar='grammar-path', type=str,
    help='path to grammar')
  parser.add_argument('--samples', metavar='N', type=int,
    help='number of samples (default: 100)', default=100)
  parser.add_argument('--start', metavar='START',
    help='start symbol of the grammar (default: A)', default='A') 
  parser.add_argument('--working-dir', metavar='wdir', dest='workdir', type=str, help='working dir, where output files are saved')
  parser.set_defaults(workdir='.') 
  parser.add_argument('--save-trees', dest='savetrees', action='store_true', help='save trees too, not just yields')
  parser.set_defaults(savetrees=True)
  parser.add_argument('--reject-smaller', metavar='rs', dest='reject_s', type=int,
    help='reject samples with yields smaller than this number', default=0)
  parser.add_argument('--reject-bigger', metavar='rb', dest='reject_b', type=int,
    help='reject samples with yields bigger than this number', default=1000000)      
  
  return parser.parse_args()
  
def main(args):
  """
  Fire up a tools and sample the requested amount of times
  """
  
  # open input file, load it into tools
  grammar_file = codecs.open(args.grammar, 'r', encoding='utf8')

  sampler = Sampler(grammar_file)
  
  # create working directory (if needed)
  if not os.path.exists(args.workdir):
    os.makedirs(args.workdir)
 
  # open output files
  out_i_yields = codecs.open("%s/%s" % (args.workdir, "out_i.yield"), 'w', encoding='utf8')
  out_o_yields = codecs.open("%s/%s" % (args.workdir, "out_o.yield"), 'w', encoding='utf8')
  
  if (args.savetrees):
    out_i_trees  = codecs.open("%s/%s" % (args.workdir, "out_i.tree"), 'w', encoding='utf8')
    out_o_trees  = codecs.open("%s/%s" % (args.workdir, "out_o.tree"), 'w', encoding='utf8')

  sample_count = 0
  
  while sample_count < args.samples:

    print("Sampling [%d/%d]..." % (sample_count + 1, args.samples))
    d = sampler.sample(args.reject_b)

    if (d == None): continue # hack to early stop generating too long sequences
    
    tree_i = sampler.derivation_to_tree(d, Nonterminal(args.start), 0)
    leaves = tree_i.leaves()
    
    # reject if yield too short or too long
    if len(leaves) < args.reject_s or len(leaves) > args.reject_b:  
      continue
    else: # sample accepted
      sample_count += 1 
      tree_o = sampler.derivation_to_tree_orhs(d, Nonterminal(args.start), 0)
 
      if(args.savetrees):
        out_i_trees.write("%s\n" % str(tree_i))
        out_o_trees.write("%s\n" % str(tree_o))
      
      out_i_yields.write("%s\n" % " ".join(leaves))
      out_o_yields.write("%s\n" % " ".join(tree_o.leaves()))
  
  # close files
  grammar_file.close()
  out_i_yields.close()
  out_o_yields.close()
  if (args.savetrees):  
    out_i_trees.close()
    out_o_trees.close()

    
if __name__ == '__main__':
  args = handle_args()
  main(args)
