"""
Toji Zenin Battle Predictor - Character Data
Prime strength stats for Toji and all major cursed spirits
"""

import pandas as pd
import numpy as np

# Toji Zenin's Prime Stats
TOJI_STATS = {
    'name': 'Toji Zenin (Fushiguro)',
    'physical_strength': 98,
    'speed': 99,
    'durability': 85,
    'combat_iq': 97,
    'cursed_energy': 0,  # Heavenly Restriction - Zero cursed energy
    'cursed_energy_resistance': 100,  # Can't be sensed, immune to barriers
    'weapon_mastery': 99,
    'stealth': 100,
    'experience': 95,
    'adaptability': 96
}

# Cursed Spirits and Sorcerers (Prime Versions)
OPPONENTS = [
    {
        'name': 'Finger Bearer',
        'grade': 'Special Grade',
        'physical_strength': 70,
        'speed': 60,
        'durability': 75,
        'cursed_energy': 80,
        'technique_power': 65,
        'intelligence': 40,
        'domain_expansion': 0,
        'regeneration': 60,
        'size_advantage': 85
    },
    {
        'name': 'Rainbow Dragon (Rika Manifestation)',
        'grade': 'Special Grade',
        'physical_strength': 85,
        'speed': 70,
        'durability': 80,
        'cursed_energy': 90,
        'technique_power': 85,
        'intelligence': 50,
        'domain_expansion': 0,
        'regeneration': 70,
        'size_advantage': 90
    },
    {
        'name': 'Dagon',
        'grade': 'Special Grade',
        'physical_strength': 75,
        'speed': 70,
        'durability': 85,
        'cursed_energy': 92,
        'technique_power': 88,
        'intelligence': 65,
        'domain_expansion': 90,  # Horizon of the Captivating Skandha
        'regeneration': 80,
        'size_advantage': 80
    },
    {
        'name': 'Hanami',
        'grade': 'Special Grade',
        'physical_strength': 80,
        'speed': 65,
        'durability': 95,  # Extremely high defense
        'cursed_energy': 90,
        'technique_power': 85,
        'intelligence': 70,
        'domain_expansion': 0,
        'regeneration': 75,
        'size_advantage': 85
    },
    {
        'name': 'Jogo',
        'grade': 'Special Grade',
        'physical_strength': 70,
        'speed': 85,
        'durability': 65,
        'cursed_energy': 95,
        'technique_power': 95,  # Devastating fire techniques
        'intelligence': 75,
        'domain_expansion': 95,  # Coffin of the Iron Mountain
        'regeneration': 70,
        'size_advantage': 60
    },
    {
        'name': 'Mahito',
        'grade': 'Special Grade',
        'physical_strength': 65,
        'speed': 80,
        'durability': 70,
        'cursed_energy': 90,
        'technique_power': 98,  # Idle Transfiguration (instant kill potential)
        'intelligence': 90,
        'domain_expansion': 92,  # Self-Embodiment of Perfection
        'regeneration': 95,  # Can reshape his soul
        'size_advantage': 50
    },
    {
        'name': 'Kurourushi (Cockroach Curse)',
        'grade': 'Special Grade',
        'physical_strength': 60,
        'speed': 75,
        'durability': 80,
        'cursed_energy': 75,
        'technique_power': 70,
        'intelligence': 55,
        'domain_expansion': 0,
        'regeneration': 90,
        'size_advantage': 70
    },
    {
        'name': 'Smallpox Deity',
        'grade': 'Special Grade',
        'physical_strength': 50,
        'speed': 60,
        'durability': 70,
        'cursed_energy': 85,
        'technique_power': 92,  # Disease manipulation
        'intelligence': 60,
        'domain_expansion': 85,
        'regeneration': 75,
        'size_advantage': 40
    },
    {
        'name': 'Naoya Zenin (Cursed Spirit)',
        'grade': 'Special Grade',
        'physical_strength': 75,
        'speed': 95,  # Projection Sorcery makes him extremely fast
        'durability': 70,
        'cursed_energy': 85,
        'technique_power': 88,
        'intelligence': 80,
        'domain_expansion': 0,
        'regeneration': 80,
        'size_advantage': 50
    },
    {
        'name': 'Geto Suguru (Prime)',
        'grade': 'Special Grade Sorcerer',
        'physical_strength': 70,
        'speed': 80,
        'durability': 75,
        'cursed_energy': 96,
        'technique_power': 94,  # Curse Manipulation (thousands of curses)
        'intelligence': 92,
        'domain_expansion': 0,
        'regeneration': 0,
        'size_advantage': 50
    },
    {
        'name': 'Gojo Satoru (Prime)',
        'grade': 'Special Grade Sorcerer',
        'physical_strength': 80,
        'speed': 95,
        'durability': 75,
        'cursed_energy': 100,
        'technique_power': 100,  # Limitless + Six Eyes
        'intelligence': 95,
        'domain_expansion': 100,  # Unlimited Void
        'regeneration': 0,
        'size_advantage': 50
    },
    {
        'name': 'Mahoraga',
        'grade': 'Special Grade Shikigami',
        'physical_strength': 95,
        'speed': 85,
        'durability': 90,
        'cursed_energy': 95,
        'technique_power': 92,
        'intelligence': 65,
        'domain_expansion': 0,
        'regeneration': 88,
        'size_advantage': 95
    },
    {
        'name': 'Sukuna (15 Fingers)',
        'grade': 'King of Curses',
        'physical_strength': 95,
        'speed': 95,
        'durability': 92,
        'cursed_energy': 98,
        'technique_power': 98,
        'intelligence': 95,
        'domain_expansion': 98,  # Malevolent Shrine
        'regeneration': 95,
        'size_advantage': 50
    },
    {
        'name': 'Sukuna (20 Fingers - Full Power)',
        'grade': 'King of Curses',
        'physical_strength': 100,
        'speed': 99,
        'durability': 98,
        'cursed_energy': 100,
        'technique_power': 100,
        'intelligence': 98,
        'domain_expansion': 100,  # Malevolent Shrine (Perfect)
        'regeneration': 99,
        'size_advantage': 50
    }
]

def create_battle_dataset():
    """
    Create training dataset with battle outcomes
    Based on canonical fights and logical power scaling
    """
    battles = []
    
    # Historical/Canonical results and extrapolations
    battle_results = [
        # Easy victories for Toji
        ('Finger Bearer', 1, 'win', 90, 'Speed and cursed tools overwhelming'),
        ('Finger Bearer', 2, 'win', 88, 'Superior tactics and weapons'),
        
        # Medium difficulty
        ('Rainbow Dragon (Rika Manifestation)', 1, 'win', 70, 'High risk, cursed tools decisive'),
        ('Dagon', 1, 'win', 85, 'Canonical - Domain means nothing to Toji'),  # Actually happened
        ('Dagon', 2, 'win', 82, 'Inverted Spear of Heaven negates domain'),
        ('Hanami', 1, 'win', 65, 'High defense challenging but beatable'),
        ('Hanami', 2, 'draw', 50, 'Extreme durability poses problems'),
        ('Kurourushi (Cockroach Curse)', 1, 'win', 75, 'Speed advantage crucial'),
        ('Naoya Zenin (Cursed Spirit)', 1, 'win', 60, 'Knows Naoya\'s techniques'),
        ('Naoya Zenin (Cursed Spirit)', 2, 'draw', 45, 'Speed vs Speed stalemate'),
        
        # High difficulty
        ('Jogo', 1, 'win', 55, 'AOE fire attacks dangerous but survivable'),
        ('Jogo', 2, 'loss', 35, 'Domain expansion + overwhelming firepower'),
        ('Smallpox Deity', 1, 'draw', 40, 'Disease attacks problematic'),
        ('Smallpox Deity', 2, 'loss', 30, 'Lacks counter to disease manipulation'),
        ('Mahito', 1, 'win', 45, 'Must avoid all physical contact'),
        ('Mahito', 2, 'draw', 35, 'One touch = death, high risk'),
        ('Mahito', 3, 'loss', 25, 'Idle Transfiguration too dangerous'),
        
        # Extreme difficulty
        ('Geto Suguru (Prime)', 1, 'win', 40, 'Numbers overwhelming but Toji assassinates'),
        ('Geto Suguru (Prime)', 2, 'loss', 30, 'Too many curses to handle'),
        ('Mahoraga', 1, 'loss', 20, 'Adaptation makes victory impossible'),
        ('Mahoraga', 2, 'loss', 15, 'Cannot be killed conventionally'),
        
        # Near impossible
        ('Gojo Satoru (Prime)', 1, 'win', 60, 'Canonical - caught off guard, no Infinity active'),  # Teen Gojo
        ('Gojo Satoru (Prime)', 2, 'loss', 5, 'Unlimited Void = instant loss'),
        ('Gojo Satoru (Prime)', 3, 'loss', 2, 'Prime Gojo unbeatable'),
        
        # Impossible
        ('Sukuna (15 Fingers)', 1, 'loss', 15, 'Overwhelming power difference'),
        ('Sukuna (15 Fingers)', 2, 'loss', 10, 'Domain guarantees death'),
        ('Sukuna (20 Fingers - Full Power)', 1, 'loss', 5, 'Complete annihilation'),
        ('Sukuna (20 Fingers - Full Power)', 2, 'loss', 2, 'No chance of survival'),
    ]
    
    for opponent_name, scenario, outcome, win_probability, notes in battle_results:
        # Find opponent stats
        opponent = next(o for o in OPPONENTS if o['name'] == opponent_name)
        
        # Calculate features
        battle = {
            # Toji advantages
            'toji_speed': TOJI_STATS['speed'],
            'toji_physical': TOJI_STATS['physical_strength'],
            'toji_combat_iq': TOJI_STATS['combat_iq'],
            'toji_stealth': TOJI_STATS['stealth'],
            'toji_weapon_mastery': TOJI_STATS['weapon_mastery'],
            'toji_ce_resistance': TOJI_STATS['cursed_energy_resistance'],
            'toji_durability': TOJI_STATS['durability'],
            
            # Opponent stats
            'opp_physical': opponent['physical_strength'],
            'opp_speed': opponent['speed'],
            'opp_durability': opponent['durability'],
            'opp_cursed_energy': opponent['cursed_energy'],
            'opp_technique_power': opponent['technique_power'],
            'opp_intelligence': opponent['intelligence'],
            'opp_domain': opponent['domain_expansion'],
            'opp_regeneration': opponent['regeneration'],
            'opp_size': opponent['size_advantage'],
            
            # Calculated advantages
            'speed_advantage': TOJI_STATS['speed'] - opponent['speed'],
            'physical_advantage': TOJI_STATS['physical_strength'] - opponent['physical_strength'],
            'iq_advantage': TOJI_STATS['combat_iq'] - opponent['intelligence'],
            'domain_nullification': TOJI_STATS['cursed_energy_resistance'] - opponent['domain_expansion'],
            
            # Danger factors
            'instant_kill_potential': 1 if opponent_name == 'Mahito' else 0,
            'aoe_attacks': 1 if opponent_name in ['Jogo', 'Smallpox Deity'] else 0,
            'adaptation_ability': 1 if opponent_name == 'Mahoraga' else 0,
            'infinity_barrier': 1 if opponent_name == 'Gojo Satoru (Prime)' else 0,
            
            # Outcome
            'outcome': outcome,
            'win_probability': win_probability,
            'opponent_name': opponent_name,
            'notes': notes
        }
        
        battles.append(battle)
    
    return pd.DataFrame(battles)

def get_opponent_by_name(name):
    """Get opponent stats by name"""
    return next((o for o in OPPONENTS if o['name'] == name), None)
