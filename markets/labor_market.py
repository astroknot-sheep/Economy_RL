"""
Labor Market - Corrected Implementation

ECONOMIC LOGIC:
- Search and matching framework (Diamond-Mortensen-Pissarides)
- Wages follow Phillips curve dynamics
- Unemployment rate emerges from matching frictions

MATCHING FUNCTION:
M = efficiency * sqrt(V * U)

Where:
- M = matches (new hires)
- V = vacancies
- U = unemployed workers
- efficiency = matching efficiency parameter

PHILLIPS CURVE:
wage_growth = -adjustment_speed * (u - u*)

Where u* is the natural unemployment rate (NAIRU)
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class LaborMarketOutcome:
    """Result of labor market clearing."""
    
    unemployment_rate: float
    employment_rate: float
    total_employed: int
    total_unemployed: int
    vacancies: int
    matches: int
    separations: int
    wage: float
    wage_growth: float
    
    # Worker allocations
    firm_workers: Dict[int, List[int]]  # firm_id -> [household_ids]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unemployment_rate": self.unemployment_rate,
            "employment_rate": self.employment_rate,
            "total_employed": self.total_employed,
            "total_unemployed": self.total_unemployed,
            "vacancies": self.vacancies,
            "matches": self.matches,
            "separations": self.separations,
            "wage": self.wage,
            "wage_growth": self.wage_growth,
        }


class LaborMarket:
    """
    Labor market with search and matching frictions.
    
    KEY MECHANISMS:
    1. Separation: Employed workers lose jobs with probability sep_rate
    2. Matching: Unemployed matched to vacancies via matching function
    3. Wages: Adjust based on unemployment gap (Phillips curve)
    """
    
    def __init__(self, config: Any):
        self.config = config
        
        # Parameters
        self.separation_rate = config.economic.separation_rate
        self.matching_efficiency = config.economic.matching_efficiency
        self.natural_unemployment = config.economic.natural_unemployment_rate
        self.wage_adjustment = config.economic.wage_adjustment_speed
        self.base_wage = config.economic.base_wage
        
        # State
        self.current_wage = self.base_wage
        self.wage_history: List[float] = [self.base_wage]
    
    def reset(self) -> None:
        """Reset labor market state."""
        self.current_wage = self.base_wage
        self.wage_history = [self.base_wage]
    
    def clear(
        self,
        households: List[Any],
        firms: List[Any],
        previous_unemployment: float = 0.05,
    ) -> LaborMarketOutcome:
        """
        Clear the labor market.
        
        SEQUENCE:
        1. Separations occur (employed → unemployed)
        2. Firms post vacancies
        3. Matching happens (unemployed → employed)
        4. Wages adjust via Phillips curve
        """
        active_households = [h for h in households if h.is_active]
        active_firms = [f for f in firms if f.is_active]
        
        if not active_households or not active_firms:
            return self._empty_outcome()
        
        n_households = len(active_households)
        
        # === 1. SEPARATIONS (BALANCED: Allow natural labor dynamics) ===
        separations = 0
        max_separations = max(5, int(len(active_households) * 0.05))  # Cap at 5% (up from 2%)
        
        for h in active_households:
            if h.state.is_employed and separations < max_separations:
                if np.random.random() < self.separation_rate:
                    h.state.is_employed = False
                    h.state.employer_id = None
                    separations += 1
        
        # === 2. COUNT STATES ===
        employed = [h for h in active_households if h.state.is_employed]
        unemployed = [h for h in active_households if not h.state.is_employed]
        
        n_unemployed = len(unemployed)
        n_employed = len(employed)
        
        # === 3. FIRM VACANCIES ===
        total_vacancies = 0
        firm_vacancies = {}
        
        for f in active_firms:
            desired = f.get_labor_demand()
            current = f.state.num_workers
            vacancy = max(0, desired - current)
            firm_vacancies[f.id] = vacancy
            total_vacancies += vacancy
        
        # === 4. MATCHING ===
        if n_unemployed > 0 and total_vacancies > 0:
            # Matching function: M = efficiency * sqrt(V * U)
            # This gives reasonable matching rates
            tightness = total_vacancies / max(n_unemployed, 1)
            
            # Job finding rate for workers
            # FIXED: Added 5% floor for frictional matching (prevents unemployment spikes)
            job_finding_rate = np.clip(
                self.matching_efficiency * np.sqrt(tightness),
                0.05,  # Minimum 5% matching rate (frictional floor)
                0.40   # Maximum 40% per month
            )
            # This floor represents job-hopping, recalls, and informal hiring
            
            # Expected matches
            expected_matches = int(n_unemployed * job_finding_rate)
            actual_matches = min(expected_matches, total_vacancies, n_unemployed)
            
            # Randomly select workers to match
            np.random.shuffle(unemployed)
            workers_to_match = unemployed[:actual_matches]
            
            # Allocate workers to firms proportionally to vacancies
            firm_ids = list(firm_vacancies.keys())
            vacancy_weights = [firm_vacancies[fid] for fid in firm_ids]
            total_weight = sum(vacancy_weights)
            
            if total_weight > 0:
                probs = [v / total_weight for v in vacancy_weights]
                
                for h in workers_to_match:
                    # Pick firm
                    firm_idx = np.random.choice(len(firm_ids), p=probs)
                    firm_id = firm_ids[firm_idx]
                    
                    # Update worker
                    h.state.is_employed = True
                    h.state.employer_id = firm_id
                    h.state.months_unemployed = 0
                    
                    # Update vacancy count
                    firm_vacancies[firm_id] = max(0, firm_vacancies[firm_id] - 1)
                    
                    # Update weights
                    vacancy_weights[firm_idx] = firm_vacancies[firm_id]
                    total_weight = sum(vacancy_weights)
                    if total_weight > 0:
                        probs = [v / total_weight for v in vacancy_weights]
                    else:
                        break
            
            matches = len(workers_to_match)
        else:
            matches = 0
        
        # === 5. BUILD FIRM WORKER LISTS ===
        firm_workers = {f.id: [] for f in active_firms}
        for h in active_households:
            if h.state.is_employed and h.state.employer_id is not None:
                if h.state.employer_id in firm_workers:
                    firm_workers[h.state.employer_id].append(h.id)
        
        # === 6. COMPUTE UNEMPLOYMENT ===
        final_employed = sum(1 for h in active_households if h.state.is_employed)
        final_unemployed = n_households - final_employed
        unemployment_rate = final_unemployed / n_households if n_households > 0 else 0
        
        # === 7. WAGE ADJUSTMENT (PHILLIPS CURVE) ===
        unemployment_gap = unemployment_rate - self.natural_unemployment
        
        # Wage growth = -adjustment * unemployment_gap
        wage_growth = -self.wage_adjustment * unemployment_gap
        wage_growth = np.clip(wage_growth, -0.02, 0.02)  # Cap at ±2% monthly
        
        new_wage = self.current_wage * (1 + wage_growth)
        new_wage = np.clip(new_wage, self.base_wage * 0.7, self.base_wage * 1.5)
        
        self.current_wage = new_wage
        self.wage_history.append(new_wage)
        
        # === 8. REMAINING VACANCIES ===
        remaining_vacancies = sum(firm_vacancies.values())
        
        return LaborMarketOutcome(
            unemployment_rate=unemployment_rate,
            employment_rate=1 - unemployment_rate,
            total_employed=final_employed,
            total_unemployed=final_unemployed,
            vacancies=remaining_vacancies,
            matches=matches,
            separations=separations,
            wage=self.current_wage,
            wage_growth=wage_growth,
            firm_workers=firm_workers,
        )
    
    def _empty_outcome(self) -> LaborMarketOutcome:
        """Return empty outcome when no agents."""
        return LaborMarketOutcome(
            unemployment_rate=1.0,
            employment_rate=0.0,
            total_employed=0,
            total_unemployed=0,
            vacancies=0,
            matches=0,
            separations=0,
            wage=self.base_wage,
            wage_growth=0.0,
            firm_workers={},
        )