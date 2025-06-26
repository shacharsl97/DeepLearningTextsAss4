import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import re
import os
import data
from data import CharTokenizer
import lm
from transformer import TransformerLM

class AttentionAnalyzer:
    """
    Comprehensive attention analysis tool for transformer language models.
    Extracts, visualizes, and analyzes attention patterns across layers and heads.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Character classification for pattern analysis
        self.vowels = set('aeiouAEIOU')
        self.consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        self.spaces = set(' ')
        self.punctuation = set('.,!?;:()[]{}"\'-')
        
        # Hebrew character sets
        self.hebrew_vowels = set('אעיהו')
        self.hebrew_consonants = set('בגדהוזסעפצקרשת')
        
    def extract_attention_matrices(self, input_text: str) -> Optional[Dict]:
        """
        Extract attention matrices for all layers and heads for given input text.
        Returns None if the input text is too short.
        """
        # Tokenize input
        tokens = self.tokenizer.tokenize(input_text)
        if len(tokens) <= 1:
            return None # Not enough context to analyze attention
        
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = self.model(input_tensor, return_attention_weights=True)
        
        # Convert to numpy for analysis and remove batch dimension
        # attention_weights shape: (n_layers, n_heads, B, N, N) -> (n_layers, n_heads, N, N)
        attention_weights = attention_weights.squeeze(2).cpu().numpy()
        
        return {
            'attention_weights': attention_weights,
            'input_tokens': tokens,
            'input_text': input_text,
            'characters': [self.tokenizer.vocab[t] for t in tokens]
        }
    
    def visualize_attention_heatmap(self, attention_data: Dict, layer: int, head: int, 
                                   save_path: Optional[str] = None, figsize: Tuple = (12, 10), title_suffix: str = ""):
        """
        Create a heatmap visualization for a specific layer and head.
        """
        attention_matrix = attention_data['attention_weights'][layer, head]
        characters = attention_data['characters']
        
        plt.figure(figsize=figsize)
        sns.heatmap(attention_matrix, 
                   xticklabels=characters, 
                   yticklabels=characters,
                   cmap='Blues',
                   cbar_kws={'label': 'Attention Weight'})
        
        title = f'Attention Heatmap - Layer {layer}, Head {head}\nInput: "{attention_data["input_text"]}"'
        if title_suffix:
            title = f'{title_suffix}\n{title}'

        plt.title(title)
        plt.xlabel('Key Position (Token being attended to)')
        plt.ylabel('Query Position (Token doing the attending)')
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved heatmap to {save_path}")
        plt.close() # Close figure to free memory

    def detect_previous_token_attention(self, attention_data: Dict) -> List[Dict]:
        """
        Calculates the 'previous token' attention score for all heads.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        results = []
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                prev_token_attention = [attention_matrix[i, i-1] for i in range(1, seq_len)]
                
                if prev_token_attention:
                    avg_prev_attention = np.mean(prev_token_attention)
                    results.append({
                        'layer': layer, 'head': head,
                        'pattern_type': 'previous_token',
                        'score': avg_prev_attention
                    })
        return results
    
    def detect_space_attention(self, attention_data: Dict) -> List[Dict]:
        """
        Calculates the 'space detection' score for all heads.
        This pattern means non-space tokens are attending to space tokens.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        characters = attention_data['characters']
        results = []
        
        space_positions = [i for i, char in enumerate(characters) if char in self.spaces]
        if not space_positions:
            return []
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                # Avg attention from non-space tokens to space tokens
                attention_to_spaces = attention_matrix[:, space_positions]
                
                # Create a mask to exclude rows that are spaces themselves
                non_space_rows_mask = [i for i, char in enumerate(characters) if char not in self.spaces]

                if not non_space_rows_mask:
                    continue # Only spaces in this text

                # Calculate score
                score = attention_to_spaces[non_space_rows_mask, :].mean()

                results.append({
                    'layer': layer, 'head': head,
                    'pattern_type': 'space_detection',
                    'score': score
                })
        return results

    def detect_vowel_consonant_patterns(self, attention_data: Dict) -> List[Dict]:
        """
        Calculates a score for vowel-consonant interaction patterns for all heads.
        A high score indicates a head treats vowels and consonants differently.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        characters = attention_data['characters']
        results = []

        is_hebrew = any(c in self.hebrew_vowels or c in self.hebrew_consonants for c in characters)
        vowel_set = self.hebrew_vowels if is_hebrew else self.vowels
        consonant_set = self.hebrew_consonants if is_hebrew else self.consonants

        vowel_indices = [i for i, c in enumerate(characters) if c in vowel_set]
        consonant_indices = [i for i, c in enumerate(characters) if c in consonant_set]

        if not vowel_indices or not consonant_indices:
            return []

        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                # Consonants attending to Vowels (excluding self-attention)
                c_to_v_attention = attention_matrix[np.ix_(consonant_indices, vowel_indices)].mean()
                
                # Vowels attending to Consonants (excluding self-attention)
                v_to_c_attention = attention_matrix[np.ix_(vowel_indices, consonant_indices)].mean()
                
                # The score is the absolute difference, showing specialization
                score = abs(c_to_v_attention - v_to_c_attention)

                results.append({
                    'layer': layer, 'head': head,
                    'pattern_type': 'vowel_consonant',
                    'score': score
                })
        return results
    
    def detect_positional_patterns(self, attention_data: Dict, max_offset: int = 5) -> List[Dict]:
        """
        Detects attention to specific relative positions (offsets).
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        results = []

        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                for offset in range(1, max_offset + 1):
                    # Attention to position i-offset from token i
                    positional_attention = [attention_matrix[i, i-offset] for i in range(offset, seq_len)]
                    if positional_attention:
                        score = np.mean(positional_attention)
                        results.append({
                            'layer': layer, 'head': head,
                            'pattern_type': f'positional_offset_-{offset}',
                            'score': score
                        })
        return results

    def generate_analysis_report(self, task_champions: Dict, language: str, output_file: str):
        """
        Generates a markdown report summarizing the best heads for each task.
        """
        report_content = [f"# Task-Based Attention Analysis Report ({language.capitalize()})\n"]
        report_content.append("This report identifies the single best attention head for several key linguistic tasks based on average scores across all samples.\n")

        for pattern_type, champion in sorted(task_champions.items()):
            score = champion['score']
            layer = champion['layer']
            head = champion['head']
            num_samples = len(champion['attention_data_list'])
            
            # Find min sequence length for the common positions info
            min_seq_len = min(attention_data['attention_weights'].shape[-1] for attention_data in champion['attention_data_list'])
            max_seq_len = max(attention_data['attention_weights'].shape[-1] for attention_data in champion['attention_data_list'])
            
            report_content.append(f"## Task: `{pattern_type}`")
            report_content.append(f"- **Champion Head**: Layer {layer}, Head {head}")
            report_content.append(f"- **Average Activation Score**: {score:.4f}")
            report_content.append(f"- **Number of Samples**: {num_samples}")
            report_content.append(f"- **Sequence Length Range**: {min_seq_len}-{max_seq_len} (averaged over common positions 1-{min_seq_len})")
            
            # Simple thresholding logic
            thresholds = {
                'previous_token': 0.5,
                'space_detection': 0.1,
                'vowel_consonant': 0.05,
                'positional_offset_-1': 0.5,
                'positional_offset_-2': 0.2,
                'positional_offset_-3': 0.1,
                'positional_offset_-4': 0.1,
                'positional_offset_-5': 0.1,
            }
            threshold = thresholds.get(pattern_type, 0.05)
            conclusion = "This head appears to be specialized for this task." if score > threshold else "This head shows weak specialization for this task."
            report_content.append(f"- **Conclusion**: {conclusion} (Threshold: {threshold})")

            # Link to the generated averaged heatmap
            avg_heatmap_filename = f"champion_head_{language}_{pattern_type}_averaged.png"
            report_content.append(f"- **Averaged Heatmap**: ![Averaged Attention Heatmap for {pattern_type}](./{avg_heatmap_filename})")
            
            # Link to individual sample heatmaps
            report_content.append(f"- **Individual Sample Heatmaps**:")
            for i in range(num_samples):
                sample_heatmap_filename = f"champion_head_{language}_{pattern_type}_sample_{i+1}.png"
                report_content.append(f"  - [Sample {i+1}](./{sample_heatmap_filename})")
            
            report_content.append("")  # Empty line for spacing

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        print(f"Generated analysis report: {output_file}")


def create_sample_texts(hebrew=False):
    if hebrew:
        return [
            "אבא קורא ספר מעניין מאוד",  # Simple sentence
            "השועל החום המהיר קופץ מעל הכלב העצלן.", # Hebrew pangram
            "אני אוהב לאכול גלידה ביום חם.", # Vowel/consonant mix
        ]
    return [
        "The quick brown fox jumps over the lazy dog.", # Pangram, good for general purpose
        "She sells seashells by the seashore.", # Repetitive sounds
        "To be or not to be, that is the question.", # Structure and punctuation
        "Peter Piper picked a peck of pickled peppers.", # Alliteration
        "io ufo aea", # Vowel-heavy
        "rhythm myths fly by", # Consonant-heavy
        "A man, a plan, a canal: Panama.", # Palindrome, structure
        "This is a test sentence with several spaces.", # Specific for space detection
    ]

def load_trained_model(checkpoint_dir: str, hebrew: bool = False, gpu: int = 0):
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Load model properties
    props_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_props.json')]
    if not props_files:
        raise FileNotFoundError(f"Could not find a properties JSON file in {checkpoint_dir}")
    props_path = os.path.join(checkpoint_dir, props_files[0])
    with open(props_path, 'r') as f:
        props = json.load(f)
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    tokenizer = CharTokenizer.load(tokenizer_path)
    
    # Init model with correct parameters
    # Use seq_len as max_context_len, and provide defaults for missing parameters
    model = TransformerLM(
        n_layers=props['n_layers'],
        n_heads=props['n_heads'],
        embed_size=props['embed_size'],
        max_context_len=props['seq_len'],  # seq_len is the context length
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=props['mlp_hidden_size'],
        with_residuals=props.get('with_residuals', True),  # Default to False if not present
        dropout=props.get('dropout', 0.0),  # Default to 0.0 if not present
        init_method=props.get('init_method', 'xavier')  # Default to xavier if not present
    ).to(device)
    
    # Load weights
    weights_path = [p for p in os.listdir(checkpoint_dir) if p.endswith('.pt')][0]
    checkpoint = torch.load(os.path.join(checkpoint_dir, weights_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer, device


def run_task_based_analysis(model, tokenizer, device, language):
    """
    Runs the full task-based analysis pipeline.
    """
    print("Starting task-based attention analysis...")
    analyzer = AttentionAnalyzer(model, tokenizer, device)
    sample_texts = create_sample_texts(hebrew=(language=='hebrew'))

    # All pattern detection functions to run
    detection_functions = {
        "previous_token": analyzer.detect_previous_token_attention,
        "space_detection": analyzer.detect_space_attention,
        "vowel_consonant": analyzer.detect_vowel_consonant_patterns,
    }
    
    # Add positional offset detectors
    for offset in range(1, 6):
        pattern_name = f'positional_offset_-{offset}'
        detection_functions[pattern_name] = lambda data, o=offset: analyzer.detect_positional_patterns(data, max_offset=o)

    # Dictionary to store all scores for each head across all samples
    all_scores = defaultdict(list)  # pattern_type -> list of (layer, head, score, attention_data)
    
    print(f"\nAnalyzing {len(sample_texts)} sample texts...")
    for i, text in enumerate(sample_texts):
        print(f"  [{i+1}/{len(sample_texts)}] Analyzing: \"{text}\"")
        attention_data = analyzer.extract_attention_matrices(text)
        if not attention_data:
            continue

        for pattern_type, func in detection_functions.items():
            # Run detection and get scores for all heads for this text
            head_scores = func(attention_data)
            
            for result in head_scores:
                # For positional patterns, we only care about the specific offset being tested
                if 'positional_offset' in pattern_type and result['pattern_type'] != pattern_type:
                    continue

                all_scores[pattern_type].append({
                    'layer': result['layer'],
                    'head': result['head'],
                    'score': result['score'],
                    'attention_data': attention_data
                })

    # Calculate average scores and find champions
    task_champions = {}
    print("\nCalculating average scores and finding champions...")
    
    for pattern_type, scores in all_scores.items():
        # Group scores by layer and head
        head_averages = defaultdict(list)
        for score_data in scores:
            key = (score_data['layer'], score_data['head'])
            head_averages[key].append(score_data['score'])
        
        # Calculate average for each head
        best_avg_score = -1
        best_head = None
        best_attention_data_list = []
        
        for (layer, head), score_list in head_averages.items():
            avg_score = np.mean(score_list)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_head = (layer, head)
                # Collect all attention data for this head
                best_attention_data_list = [s['attention_data'] for s in scores 
                                          if s['layer'] == layer and s['head'] == head]
        
        if best_head:
            task_champions[pattern_type] = {
                'layer': best_head[0],
                'head': best_head[1],
                'score': best_avg_score,
                'pattern_type': pattern_type,
                'attention_data_list': best_attention_data_list
            }

    print("\nAnalysis complete. Found champions for the following tasks:")
    for pattern_type, champ in task_champions.items():
        print(f"  - {pattern_type}: Layer {champ['layer']}, Head {champ['head']} (Avg Score: {champ['score']:.4f})")

    print("\nGenerating heatmaps for champion heads...")
    for pattern_type, champ in task_champions.items():
        layer, head = champ['layer'], champ['head']
        attention_data_list = champ['attention_data_list']
        
        # Generate individual heatmaps for each sample
        for i, attention_data in enumerate(attention_data_list):
            sample_heatmap_filename = f"champion_head_{language}_{pattern_type}_sample_{i+1}.png"
            analyzer.visualize_attention_heatmap(
                attention_data=attention_data,
                layer=layer,
                head=head,
                save_path=os.path.join('search_results', sample_heatmap_filename),
                title_suffix=f"Champion Head for {pattern_type.replace('_', ' ').title()} - Sample {i+1}"
            )
        
        # Generate averaged heatmap
        if attention_data_list:
            # Find the minimum sequence length across all samples
            min_seq_len = min(attention_data['attention_weights'].shape[-1] for attention_data in attention_data_list)
            
            # Average only the common positions (up to min_seq_len)
            avg_attention_matrix = np.zeros((attention_data_list[0]['attention_weights'].shape[0],  # n_layers
                                           attention_data_list[0]['attention_weights'].shape[1],   # n_heads
                                           min_seq_len, min_seq_len))
            
            for attention_data in attention_data_list:
                # Take only the first min_seq_len positions
                truncated_attention = attention_data['attention_weights'][:, :, :min_seq_len, :min_seq_len]
                avg_attention_matrix += truncated_attention
            
            avg_attention_matrix /= len(attention_data_list)
            
            # Create averaged attention data using the shortest sample's tokens/characters (truncated)
            shortest_sample = min(attention_data_list, key=lambda x: len(x['characters']))
            avg_attention_data = {
                'attention_weights': avg_attention_matrix,
                'input_tokens': shortest_sample['input_tokens'][:min_seq_len],
                'input_text': f"Average across {len(attention_data_list)} samples (common positions 1-{min_seq_len})",
                'characters': shortest_sample['characters'][:min_seq_len]
            }
            
            avg_heatmap_filename = f"champion_head_{language}_{pattern_type}_averaged.png"
            analyzer.visualize_attention_heatmap(
                attention_data=avg_attention_data,
                layer=layer,
                head=head,
                save_path=os.path.join('search_results', avg_heatmap_filename),
                title_suffix=f"Champion Head for {pattern_type.replace('_', ' ').title()} - Averaged (Common Positions)"
            )
    
    # Generate the final report
    report_path = f"task_based_attention_report_{language}.md"
    analyzer.generate_analysis_report(task_champions, language, os.path.join('search_results', report_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run task-based attention analysis on a transformer model.")
    parser.add_argument('language', type=str, choices=['english', 'hebrew'], help="Language of the model to analyze.")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Path to a specific checkpoint directory. If not provided, the latest will be used.")
    args = parser.parse_args()

    if args.checkpoint_dir:
        final_checkpoint_dir = args.checkpoint_dir
    else:
        # Find the latest checkpoint for the chosen language
        checkpoint_base_dir = 'checkpoints'
        dirs = [d for d in os.listdir(checkpoint_base_dir) if d.startswith(f'{args.language}_step_')]
        if not dirs:
            raise FileNotFoundError(f"No checkpoint directories found for language '{args.language}' in '{checkpoint_base_dir}'")

        latest_step = -1
        latest_dir = ''
        for d in dirs:
            try:
                step = int(d.split('_')[-1])
                if step > latest_step:
                    latest_step = step
                    latest_dir = d
            except (ValueError, IndexError):
                continue
        
        final_checkpoint_dir = os.path.join(checkpoint_base_dir, latest_dir)

    print(f"Using checkpoint: {final_checkpoint_dir}")
    model, tokenizer, device = load_trained_model(final_checkpoint_dir, hebrew=(args.language=='hebrew'))
    
    # Make sure search_results directory exists
    if not os.path.exists('search_results'):
        os.makedirs('search_results')
        
    run_task_based_analysis(model, tokenizer, device, args.language)
    print("\nDone.") 