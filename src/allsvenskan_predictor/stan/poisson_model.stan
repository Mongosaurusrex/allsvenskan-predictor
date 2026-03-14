data {

  int<lower=1> N;                // matches
  int<lower=1> T;                // teams

  array[N] int<lower=1, upper=T> home_team;
  array[N] int<lower=1, upper=T> away_team;

  array[N] int<lower=0> home_goals;
  array[N] int<lower=0> away_goals;

  array[N] real weights;         // time-decay weights
}

parameters {

  vector[T] attack_raw;
  vector[T] defense_raw;

  real<lower=0> sigma_attack;
  real<lower=0> sigma_defense;

  real gamma;
}

transformed parameters {

  vector[T] attack;
  vector[T] defense;

  attack = attack_raw - mean(attack_raw);
  defense = defense_raw - mean(defense_raw);
}

model {

  sigma_attack ~ normal(0,0.5);
  sigma_defense ~ normal(0,0.5);

  attack_raw ~ normal(0, sigma_attack);
  defense_raw ~ normal(0, sigma_defense);

  gamma ~ normal(0,0.5);

  for (n in 1:N) {

    real log_lambda_home =
      attack[home_team[n]]
      - defense[away_team[n]]
      + gamma;

    real log_lambda_away =
      attack[away_team[n]]
      - defense[home_team[n]];

    target += weights[n] *
      poisson_log_lpmf(home_goals[n] | log_lambda_home);

    target += weights[n] *
      poisson_log_lpmf(away_goals[n] | log_lambda_away);
  }
}