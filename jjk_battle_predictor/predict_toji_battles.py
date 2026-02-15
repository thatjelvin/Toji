"""
Main script to predict Toji's battles against all cursed spirits
"""

from toji_battle_data import create_battle_dataset, OPPONENTS, TOJI_STATS, get_opponent_by_name
from toji_ml_model import TojiBattlePredictor
import pandas as pd

def create_battle_features(opponent):
    """Create feature dictionary for a battle against an opponent"""
    return {
        'toji_speed': TOJI_STATS['speed'],
        'toji_physical': TOJI_STATS['physical_strength'],
        'toji_combat_iq': TOJI_STATS['combat_iq'],
        'toji_stealth': TOJI_STATS['stealth'],
        'toji_weapon_mastery': TOJI_STATS['weapon_mastery'],
        'toji_ce_resistance': TOJI_STATS['cursed_energy_resistance'],
        'toji_durability': TOJI_STATS['durability'],
        'opp_physical': opponent['physical_strength'],
        'opp_speed': opponent['speed'],
        'opp_durability': opponent['durability'],
        'opp_cursed_energy': opponent['cursed_energy'],
        'opp_technique_power': opponent['technique_power'],
        'opp_intelligence': opponent['intelligence'],
        'opp_domain': opponent['domain_expansion'],
        'opp_regeneration': opponent['regeneration'],
        'opp_size': opponent['size_advantage'],
        'speed_advantage': TOJI_STATS['speed'] - opponent['speed'],
        'physical_advantage': TOJI_STATS['physical_strength'] - opponent['physical_strength'],
        'iq_advantage': TOJI_STATS['combat_iq'] - opponent['intelligence'],
        'domain_nullification': TOJI_STATS['cursed_energy_resistance'] - opponent['domain_expansion'],
        'instant_kill_potential': 1 if opponent['name'] == 'Mahito' else 0,
        'aoe_attacks': 1 if opponent['name'] in ['Jogo', 'Smallpox Deity'] else 0,
        'adaptation_ability': 1 if opponent['name'] == 'Mahoraga' else 0,
        'infinity_barrier': 1 if opponent['name'] == 'Gojo Satoru (Prime)' else 0,
    }

def predict_all_battles():
    """Train model and predict all battles"""
    print("\n" + "=" * 80)
    print(" " * 20 + "TOJI ZENIN BATTLE SIMULATOR")
    print(" " * 15 + "The Sorcerer Killer vs All Major Threats")
    print("=" * 80)
    
    # Load and prepare data
    print("\n📊 Loading battle dataset...")
    df = create_battle_dataset()
    print(f"✅ Loaded {len(df)} training battles")
    
    # Train model
    predictor = TojiBattlePredictor()
    predictor.train(df)
    
    # Save model
    predictor.save_model('jjk_battle_predictor/toji_battle_model.pkl')
    
    # Predict battles against all opponents
    print("\n" + "=" * 80)
    print("BATTLE PREDICTIONS - TOJI AT PRIME STRENGTH")
    print("=" * 80)
    
    results = []
    
    for opponent in OPPONENTS:
        print(f"\n{'─' * 80}")
        print(f"⚔️  TOJI ZENIN vs {opponent['name'].upper()}")
        print(f"{'─' * 80}")
        print(f"Grade: {opponent['grade']}")
        
        # Create battle features
        battle_features = create_battle_features(opponent)
        
        # Predict
        prediction = predictor.predict_battle(battle_features)
        
        # Display results
        outcome = prediction['predicted_outcome']
        outcome_emoji = {'win': '✅', 'draw': '⚡', 'loss': '❌'}
        
        print(f"\n🎯 Predicted Outcome: {outcome_emoji[outcome]} {outcome.upper()}")
        print(f"\n📈 Probability Breakdown:")
        print(f"   Win:  {prediction['win_chance']:5.1f}% {'█' * int(prediction['win_chance'] / 5)}")
        print(f"   Draw: {prediction['draw_chance']:5.1f}% {'█' * int(prediction['draw_chance'] / 5)}")
        print(f"   Loss: {prediction['loss_chance']:5.1f}% {'█' * int(prediction['loss_chance'] / 5)}")
        
        # Key factors
        print(f"\n🔍 Key Battle Factors:")
        print(f"   Speed Differential: {battle_features['speed_advantage']:+d}")
        print(f"   Physical Advantage: {battle_features['physical_advantage']:+d}")
        print(f"   Combat IQ Edge: {battle_features['iq_advantage']:+d}")
        print(f"   Domain Nullification: {battle_features['domain_nullification']:+d}")
        
        # Special factors
        special_factors = []
        if opponent['domain_expansion'] > 0:
            special_factors.append(f"⚠️  Domain Expansion ({opponent['domain_expansion']})")
        if opponent['regeneration'] > 80:
            special_factors.append(f"⚠️  High Regeneration ({opponent['regeneration']})")
        if battle_features['instant_kill_potential']:
            special_factors.append("☠️  INSTANT KILL POTENTIAL (Idle Transfiguration)")
        if battle_features['infinity_barrier']:
            special_factors.append("🛡️  Infinity Barrier (Near-Impenetrable)")
        if battle_features['adaptation_ability']:
            special_factors.append("🔄 Adaptation Ability (Mahoraga)")
        
        if special_factors:
            print(f"\n⚠️  Special Threat Factors:")
            for factor in special_factors:
                print(f"   {factor}")
        
        # Strategic analysis
        print(f"\n💭 Strategic Analysis:")
        if outcome == 'win':
            if prediction['win_chance'] > 75:
                print("   Toji has overwhelming advantage. Speed and cursed tools dominate.")
            elif prediction['win_chance'] > 60:
                print("   Toji likely wins but must avoid mistakes. Calculated approach needed.")
            else:
                print("   High-risk victory. Toji must exploit weaknesses perfectly.")
        elif outcome == 'draw':
            print("   Stalemate scenario. Neither can secure decisive victory.")
        else:
            if prediction['loss_chance'] > 80:
                print("   Overwhelming power difference. Toji cannot overcome this opponent.")
            else:
                print("   Extremely dangerous. Toji's survival chances are low.")
        
        results.append({
            'Opponent': opponent['name'],
            'Grade': opponent['grade'],
            'Predicted': outcome,
            'Win %': f"{prediction['win_chance']:.1f}%",
            'Draw %': f"{prediction['draw_chance']:.1f}%",
            'Loss %': f"{prediction['loss_chance']:.1f}%"
        })
    
    # Summary table
    print("\n" + "=" * 80)
    print("BATTLE SUMMARY - ALL MATCHUPS")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Final analysis
    wins = sum(1 for r in results if r['Predicted'] == 'win')
    draws = sum(1 for r in results if r['Predicted'] == 'draw')
    losses = sum(1 for r in results if r['Predicted'] == 'loss')
    
    print(f"\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"Wins:  {wins}/{len(results)}")
    print(f"Draws: {draws}/{len(results)}")
    print(f"Losses: {losses}/{len(results)}")
    print(f"\nWin Rate: {wins/len(results)*100:.1f}%")
    
    print("\n💬 Conclusion:")
    print("   Toji Zenin is an exceptional fighter who can defeat most Special Grade")
    print("   curses thanks to his Heavenly Restriction, combat genius, and cursed tools.")
    print("   However, certain abilities (Idle Transfiguration, Infinity, Sukuna's power)")
    print("   represent insurmountable obstacles even for the Sorcerer Killer.")
    print("   His greatest asset is his complete invisibility to cursed energy detection,")
    print("   allowing him to assassinate targets before they can react.")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    predict_all_battles()
