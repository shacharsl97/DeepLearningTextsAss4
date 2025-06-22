import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import re

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
        self.hebrew_consonants = set('בגדהוזחטסעפצקרשת')
        
    def extract_attention_matrices(self, input_text: str) -> Dict:
        """
        Extract attention matrices for all layers and heads for given input text.
        
        Returns:
            Dict containing:
            - 'attention_weights': (n_layers, n_heads, seq_len, seq_len)
            - 'input_tokens': List of character tokens
            - 'input_text': Original input text
        """
        # Tokenize input
        tokens = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = self.model(input_tensor, return_attention_weights=True)
        
        # Convert to numpy for analysis
        attention_weights = attention_weights.cpu().numpy()
        
        return {
            'attention_weights': attention_weights,
            'input_tokens': tokens,
            'input_text': input_text,
            'characters': [self.tokenizer.decode([t]) for t in tokens]
        }
    
    def visualize_attention_heatmap(self, attention_data: Dict, layer: int, head: int, 
                                   save_path: Optional[str] = None, figsize: Tuple = (10, 8)):
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
        
        plt.title(f'Attention Heatmap - Layer {layer}, Head {head}\nInput: "{attention_data["input_text"]}"')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def visualize_multi_head_attention(self, attention_data: Dict, layer: int, 
                                      save_path: Optional[str] = None):
        """
        Visualize all heads in a specific layer.
        """
        n_heads = attention_data['attention_weights'].shape[1]
        fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_heads > 1 else [axes]
        
        for head in range(n_heads):
            attention_matrix = attention_data['attention_weights'][layer, head]
            characters = attention_data['characters']
            
            sns.heatmap(attention_matrix, 
                       xticklabels=characters, 
                       yticklabels=characters,
                       cmap='Blues',
                       ax=axes[head],
                       cbar=False)
            axes[head].set_title(f'Head {head}')
            axes[head].set_xlabel('Key')
            axes[head].set_ylabel('Query')
        
        # Hide unused subplots
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'All Attention Heads - Layer {layer}\nInput: "{attention_data["input_text"]}"')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def detect_previous_token_attention(self, attention_data: Dict, threshold: float = 0.5) -> Dict:
        """
        Detect heads that consistently attend to the previous token.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        results = {}
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                # Check diagonal-1 (previous token attention)
                prev_token_attention = []
                for i in range(1, seq_len):
                    prev_token_attention.append(attention_matrix[i, i-1])
                
                avg_prev_attention = np.mean(prev_token_attention)
                max_prev_attention = np.max(prev_token_attention)
                
                if avg_prev_attention > threshold:
                    results[f'layer_{layer}_head_{head}'] = {
                        'avg_prev_attention': avg_prev_attention,
                        'max_prev_attention': max_prev_attention,
                        'pattern_strength': avg_prev_attention,
                        'pattern_type': 'previous_token'
                    }
        
        return results
    
    def detect_space_attention(self, attention_data: Dict, threshold: float = 0.3) -> Dict:
        """
        Detect heads that attend to space characters.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        characters = attention_data['characters']
        results = {}
        
        # Find space positions
        space_positions = [i for i, char in enumerate(characters) if char == ' ']
        
        if not space_positions:
            return results
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                # Calculate average attention to spaces
                space_attention_scores = []
                for pos in range(seq_len):
                    if pos not in space_positions:  # Don't count spaces attending to themselves
                        attention_to_spaces = [attention_matrix[pos, space_pos] for space_pos in space_positions]
                        if attention_to_spaces:
                            space_attention_scores.append(np.mean(attention_to_spaces))
                
                if space_attention_scores:
                    avg_space_attention = np.mean(space_attention_scores)
                    if avg_space_attention > threshold:
                        results[f'layer_{layer}_head_{head}'] = {
                            'avg_space_attention': avg_space_attention,
                            'pattern_strength': avg_space_attention,
                            'pattern_type': 'space_detection'
                        }
        
        return results
    
    def detect_vowel_consonant_patterns(self, attention_data: Dict, threshold: float = 0.2) -> Dict:
        """
        Detect heads that show different attention patterns for vowels vs consonants.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        characters = attention_data['characters']
        results = {}
        
        # Classify characters
        vowel_positions = []
        consonant_positions = []
        
        for i, char in enumerate(characters):
            if char in self.vowels or char in self.hebrew_vowels:
                vowel_positions.append(i)
            elif char in self.consonants or char in self.hebrew_consonants:
                consonant_positions.append(i)
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                # Calculate attention patterns for vowels and consonants
                vowel_attention_patterns = []
                consonant_attention_patterns = []
                
                # Vowels attending to consonants
                for v_pos in vowel_positions:
                    for c_pos in consonant_positions:
                        if v_pos != c_pos:
                            vowel_attention_patterns.append(attention_matrix[v_pos, c_pos])
                
                # Consonants attending to vowels
                for c_pos in consonant_positions:
                    for v_pos in vowel_positions:
                        if c_pos != v_pos:
                            consonant_attention_patterns.append(attention_matrix[c_pos, v_pos])
                
                if vowel_attention_patterns and consonant_attention_patterns:
                    avg_vowel_to_consonant = np.mean(vowel_attention_patterns)
                    avg_consonant_to_vowel = np.mean(consonant_attention_patterns)
                    
                    # Check if there's a significant difference
                    pattern_strength = abs(avg_vowel_to_consonant - avg_consonant_to_vowel)
                    
                    if pattern_strength > threshold:
                        results[f'layer_{layer}_head_{head}'] = {
                            'vowel_to_consonant': avg_vowel_to_consonant,
                            'consonant_to_vowel': avg_consonant_to_vowel,
                            'pattern_strength': pattern_strength,
                            'pattern_type': 'vowel_consonant'
                        }
        
        return results
    
    def detect_positional_patterns(self, attention_data: Dict, max_offset: int = 5, threshold: float = 0.3) -> Dict:
        """
        Detect heads that attend to specific relative positions.
        """
        n_layers, n_heads, seq_len, _ = attention_data['attention_weights'].shape
        results = {}
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attention_matrix = attention_data['attention_weights'][layer, head]
                
                # Check different positional offsets
                for offset in range(-max_offset, max_offset + 1):
                    if offset == 0:
                        continue
                    
                    offset_attention_scores = []
                    for i in range(seq_len):
                        j = i + offset
                        if 0 <= j < seq_len:
                            offset_attention_scores.append(attention_matrix[i, j])
                    
                    if offset_attention_scores:
                        avg_offset_attention = np.mean(offset_attention_scores)
                        if avg_offset_attention > threshold:
                            results[f'layer_{layer}_head_{head}_offset_{offset}'] = {
                                'offset': offset,
                                'avg_attention': avg_offset_attention,
                                'pattern_strength': avg_offset_attention,
                                'pattern_type': f'positional_offset_{offset}'
                            }
        
        return results
    
    def analyze_attention_patterns(self, input_texts: List[str]) -> Dict:
        """
        Comprehensive analysis of attention patterns across multiple inputs.
        """
        all_results = {
            'previous_token': defaultdict(list),
            'space_detection': defaultdict(list),
            'vowel_consonant': defaultdict(list),
            'positional': defaultdict(list)
        }
        
        for i, text in enumerate(input_texts):
            print(f"Analyzing text {i+1}/{len(input_texts)}: '{text[:50]}...'")
            
            attention_data = self.extract_attention_matrices(text)
            
            # Detect different patterns
            prev_token_results = self.detect_previous_token_attention(attention_data)
            space_results = self.detect_space_attention(attention_data)
            vowel_consonant_results = self.detect_vowel_consonant_patterns(attention_data)
            positional_results = self.detect_positional_patterns(attention_data)
            
            # Aggregate results
            for pattern_type, results in [
                ('previous_token', prev_token_results),
                ('space_detection', space_results),
                ('vowel_consonant', vowel_consonant_results),
                ('positional', positional_results)
            ]:
                for head_key, result in results.items():
                    all_results[pattern_type][head_key].append(result)
        
        # Calculate consistency scores
        final_results = {}
        for pattern_type, head_results in all_results.items():
            for head_key, results in head_results.items():
                if len(results) >= 2:  # Need at least 2 samples for consistency
                    avg_strength = np.mean([r['pattern_strength'] for r in results])
                    consistency = 1.0 - np.std([r['pattern_strength'] for r in results]) / avg_strength
                    
                    final_results[f"{pattern_type}_{head_key}"] = {
                        'pattern_type': pattern_type,
                        'head_key': head_key,
                        'avg_strength': avg_strength,
                        'consistency': consistency,
                        'num_samples': len(results),
                        'details': results
                    }
        
        return final_results
    
    def find_most_interpretable_heads(self, analysis_results: Dict, min_consistency: float = 0.7) -> List[Dict]:
        """
        Find the most interpretable attention heads based on pattern strength and consistency.
        """
        interpretable_heads = []
        
        for head_key, result in analysis_results.items():
            if result['consistency'] >= min_consistency and result['avg_strength'] > 0.3:
                interpretable_heads.append({
                    'head_key': head_key,
                    'pattern_type': result['pattern_type'],
                    'avg_strength': result['avg_strength'],
                    'consistency': result['consistency'],
                    'interpretability_score': result['avg_strength'] * result['consistency']
                })
        
        # Sort by interpretability score
        interpretable_heads.sort(key=lambda x: x['interpretability_score'], reverse=True)
        return interpretable_heads
    
    def generate_analysis_report(self, analysis_results: Dict, interpretable_heads: List[Dict], 
                                output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive analysis report.
        """
        report = []
        report.append("# Attention Pattern Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"- Total patterns analyzed: {len(analysis_results)}")
        report.append(f"- Interpretable heads found: {len(interpretable_heads)}")
        report.append("")
        
        # Most interpretable heads
        report.append("## Most Interpretable Attention Heads")
        report.append("")
        
        for i, head in enumerate(interpretable_heads[:10]):  # Top 10
            report.append(f"### {i+1}. {head['head_key']}")
            report.append(f"- **Pattern Type**: {head['pattern_type']}")
            report.append(f"- **Average Strength**: {head['avg_strength']:.3f}")
            report.append(f"- **Consistency**: {head['consistency']:.3f}")
            report.append(f"- **Interpretability Score**: {head['interpretability_score']:.3f}")
            report.append("")
        
        # Pattern type breakdown
        report.append("## Pattern Type Breakdown")
        pattern_counts = defaultdict(int)
        for head in interpretable_heads:
            pattern_counts[head['pattern_type']] += 1
        
        for pattern_type, count in pattern_counts.items():
            report.append(f"- **{pattern_type}**: {count} heads")
        report.append("")
        
        # Detailed analysis
        report.append("## Detailed Analysis")
        for head_key, result in analysis_results.items():
            if result['consistency'] > 0.5:  # Show moderately consistent patterns
                report.append(f"### {head_key}")
                report.append(f"- Pattern Type: {result['pattern_type']}")
                report.append(f"- Average Strength: {result['avg_strength']:.3f}")
                report.append(f"- Consistency: {result['consistency']:.3f}")
                report.append(f"- Samples: {result['num_samples']}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_attention_data(self, attention_data: Dict, filename: str):
        """
        Save attention data to file for later analysis.
        """
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {
            'attention_weights': attention_data['attention_weights'].tolist(),
            'input_tokens': attention_data['input_tokens'],
            'input_text': attention_data['input_text'],
            'characters': attention_data['characters']
        }
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def load_attention_data(self, filename: str) -> Dict:
        """
        Load attention data from file.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        data['attention_weights'] = np.array(data['attention_weights'])
        return data


def create_sample_texts(hebrew=False):
    """
    Create sample texts for analysis.
    """
    if hebrew:
        return [
            "שלום עולם",
            "בוקר טוב לכולם",
            "אני אוהב ללמוד",
            "השמש זורחת בבוקר",
            "הספר נמצא על השולחן"
        ]
    else:
        return [
            "Hello world",
            "Good morning everyone",
            "I love learning",
            "The sun rises in the morning",
            "The book is on the table",
            "Attention patterns are fascinating",
            "Transformers process text efficiently",
            "Machine learning models learn patterns"
        ]


# Example usage functions
def run_attention_analysis(model, tokenizer, hebrew=False, save_visualizations=True):
    """
    Run complete attention analysis pipeline.
    """
    analyzer = AttentionAnalyzer(model, tokenizer)
    
    # Get sample texts
    sample_texts = create_sample_texts(hebrew)
    
    # Run analysis
    print("Running attention pattern analysis...")
    analysis_results = analyzer.analyze_attention_patterns(sample_texts)
    
    # Find interpretable heads
    interpretable_heads = analyzer.find_most_interpretable_heads(analysis_results)
    
    # Generate report
    report = analyzer.generate_analysis_report(analysis_results, interpretable_heads)
    print(report)
    
    # Save report
    lang_suffix = "hebrew" if hebrew else "english"
    with open(f"attention_analysis_report_{lang_suffix}.md", 'w') as f:
        f.write(report)
    
    # Create visualizations for top interpretable heads
    if save_visualizations and interpretable_heads:
        top_head = interpretable_heads[0]
        head_key = top_head['head_key']
        
        # Parse layer and head from key
        if 'layer_' in head_key and 'head_' in head_key:
            parts = head_key.split('_')
            layer = int(parts[1])
            head = int(parts[3])
            
            # Create visualization for first sample text
            attention_data = analyzer.extract_attention_matrices(sample_texts[0])
            analyzer.visualize_attention_heatmap(
                attention_data, layer, head, 
                save_path=f"attention_heatmap_{lang_suffix}_layer{layer}_head{head}.png"
            )
    
    return analysis_results, interpretable_heads 