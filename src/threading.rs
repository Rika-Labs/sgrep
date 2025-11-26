use std::sync::OnceLock;

pub struct ThreadConfig {
    pub rayon_threads: usize,
    pub onnx_threads: usize,
    pub walker_threads: usize,
}

#[derive(Clone, Debug, Default)]
pub enum CpuPreset {
    #[default]
    Auto,
    Low,
    Medium,
    High,
    Background,
}

static CONFIG: OnceLock<ThreadConfig> = OnceLock::new();

impl Default for ThreadConfig {
    fn default() -> Self {
        Self::compute(None, None)
    }
}

impl ThreadConfig {
    pub fn get() -> &'static ThreadConfig {
        CONFIG.get_or_init(Self::default)
    }

    pub fn init(max_threads: Option<usize>, preset: Option<CpuPreset>) {
        let _ = CONFIG.set(Self::compute(max_threads, preset));
    }

    fn compute(max_threads: Option<usize>, preset: Option<CpuPreset>) -> Self {
        let total_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        let preset = preset.unwrap_or_default();
        let percent = match preset {
            CpuPreset::Auto => 75,
            CpuPreset::Low | CpuPreset::Background => 25,
            CpuPreset::Medium => 50,
            CpuPreset::High => 100,
        };

        let budget = max_threads
            .filter(|&t| t > 0)
            .unwrap_or((total_cores * percent) / 100)
            .max(2);

        let onnx_threads = (budget / 4).clamp(2, 4);
        let rayon_threads = budget.saturating_sub(onnx_threads / 2).max(2);
        let walker_threads = rayon_threads.min(8);

        Self {
            rayon_threads,
            onnx_threads,
            walker_threads,
        }
    }

    pub fn apply(&self) {
        use std::env;

        if env::var_os("RAYON_NUM_THREADS").is_none() {
            env::set_var("RAYON_NUM_THREADS", self.rayon_threads.to_string());
        }

        if env::var_os("ORT_NUM_THREADS").is_none() {
            env::set_var("ORT_NUM_THREADS", self.onnx_threads.to_string());
        }

        if env::var_os("ORT_INTER_OP_NUM_THREADS").is_none() {
            env::set_var("ORT_INTER_OP_NUM_THREADS", "1");
        }
    }
}

impl CpuPreset {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "background" => Some(Self::Background),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;

    #[test]
    fn preset_from_str_parses_valid_values() {
        assert!(matches!(CpuPreset::from_str("auto"), Some(CpuPreset::Auto)));
        assert!(matches!(CpuPreset::from_str("LOW"), Some(CpuPreset::Low)));
        assert!(matches!(
            CpuPreset::from_str("Medium"),
            Some(CpuPreset::Medium)
        ));
        assert!(matches!(CpuPreset::from_str("HIGH"), Some(CpuPreset::High)));
        assert!(matches!(
            CpuPreset::from_str("background"),
            Some(CpuPreset::Background)
        ));
    }

    #[test]
    fn preset_from_str_returns_none_for_invalid() {
        assert!(CpuPreset::from_str("invalid").is_none());
        assert!(CpuPreset::from_str("").is_none());
        assert!(CpuPreset::from_str("EXTREME").is_none());
        assert!(CpuPreset::from_str("123").is_none());
    }

    #[test]
    fn preset_from_str_is_case_insensitive() {
        assert!(CpuPreset::from_str("AUTO").is_some());
        assert!(CpuPreset::from_str("Auto").is_some());
        assert!(CpuPreset::from_str("auto").is_some());
        assert!(CpuPreset::from_str("AuTo").is_some());
    }

    #[test]
    fn thread_config_respects_minimum_threads() {
        let config = ThreadConfig::compute(Some(1), None);
        assert!(config.rayon_threads >= 2);
        assert!(config.onnx_threads >= 2);
        assert!(config.walker_threads >= 1);
    }

    #[test]
    fn thread_config_zero_means_auto() {
        let auto = ThreadConfig::compute(None, Some(CpuPreset::Auto));
        let zero = ThreadConfig::compute(Some(0), Some(CpuPreset::Auto));
        assert_eq!(auto.rayon_threads, zero.rayon_threads);
        assert_eq!(auto.onnx_threads, zero.onnx_threads);
    }

    #[test]
    fn thread_config_respects_preset_percentages() {
        let high = ThreadConfig::compute(None, Some(CpuPreset::High));
        let low = ThreadConfig::compute(None, Some(CpuPreset::Low));
        assert!(
            high.rayon_threads >= low.rayon_threads,
            "high ({}) should be >= low ({})",
            high.rayon_threads,
            low.rayon_threads
        );
    }

    #[test]
    fn thread_config_low_and_background_are_equivalent() {
        let low = ThreadConfig::compute(Some(16), Some(CpuPreset::Low));
        let background = ThreadConfig::compute(Some(16), Some(CpuPreset::Background));
        assert_eq!(low.rayon_threads, background.rayon_threads);
        assert_eq!(low.onnx_threads, background.onnx_threads);
        assert_eq!(low.walker_threads, background.walker_threads);
    }

    #[test]
    fn walker_threads_capped_at_8() {
        let config = ThreadConfig::compute(Some(32), Some(CpuPreset::High));
        assert!(config.walker_threads <= 8);

        let config = ThreadConfig::compute(Some(64), Some(CpuPreset::High));
        assert!(config.walker_threads <= 8);
    }

    #[test]
    fn onnx_threads_capped_at_4() {
        let config = ThreadConfig::compute(Some(32), Some(CpuPreset::High));
        assert!(config.onnx_threads <= 4);
        assert!(config.onnx_threads >= 2);
    }

    #[test]
    fn thread_config_explicit_max_overrides_preset() {
        let config = ThreadConfig::compute(Some(4), Some(CpuPreset::High));
        assert!(config.rayon_threads <= 4);
    }

    #[test]
    fn thread_config_handles_small_budgets() {
        let config = ThreadConfig::compute(Some(2), None);
        assert!(config.rayon_threads >= 2);
        assert!(config.onnx_threads >= 2);

        let config = ThreadConfig::compute(Some(3), None);
        assert!(config.rayon_threads >= 2);
        assert!(config.onnx_threads >= 2);
    }

    #[test]
    fn thread_config_default_is_auto_preset() {
        let default = ThreadConfig::default();
        let auto = ThreadConfig::compute(None, Some(CpuPreset::Auto));
        assert_eq!(default.rayon_threads, auto.rayon_threads);
    }

    #[test]
    fn preset_default_is_auto() {
        let preset: CpuPreset = Default::default();
        assert!(matches!(preset, CpuPreset::Auto));
    }

    #[test]
    fn thread_config_medium_is_between_low_and_high() {
        let low = ThreadConfig::compute(Some(16), Some(CpuPreset::Low));
        let medium = ThreadConfig::compute(Some(16), Some(CpuPreset::Medium));
        let high = ThreadConfig::compute(Some(16), Some(CpuPreset::High));

        assert!(low.rayon_threads <= medium.rayon_threads);
        assert!(medium.rayon_threads <= high.rayon_threads);
    }

    #[test]
    #[serial]
    fn apply_sets_rayon_env_var() {
        let prev = env::var("RAYON_NUM_THREADS").ok();
        env::remove_var("RAYON_NUM_THREADS");

        let config = ThreadConfig::compute(Some(6), None);
        config.apply();

        let val = env::var("RAYON_NUM_THREADS").unwrap();
        assert_eq!(val, config.rayon_threads.to_string());

        if let Some(v) = prev {
            env::set_var("RAYON_NUM_THREADS", v);
        } else {
            env::remove_var("RAYON_NUM_THREADS");
        }
    }

    #[test]
    #[serial]
    fn apply_sets_ort_env_vars() {
        let prev_intra = env::var("ORT_NUM_THREADS").ok();
        let prev_inter = env::var("ORT_INTER_OP_NUM_THREADS").ok();
        env::remove_var("ORT_NUM_THREADS");
        env::remove_var("ORT_INTER_OP_NUM_THREADS");

        let config = ThreadConfig::compute(Some(8), None);
        config.apply();

        let intra = env::var("ORT_NUM_THREADS").unwrap();
        let inter = env::var("ORT_INTER_OP_NUM_THREADS").unwrap();
        assert_eq!(intra, config.onnx_threads.to_string());
        assert_eq!(inter, "1");

        if let Some(v) = prev_intra {
            env::set_var("ORT_NUM_THREADS", v);
        } else {
            env::remove_var("ORT_NUM_THREADS");
        }
        if let Some(v) = prev_inter {
            env::set_var("ORT_INTER_OP_NUM_THREADS", v);
        } else {
            env::remove_var("ORT_INTER_OP_NUM_THREADS");
        }
    }

    #[test]
    #[serial]
    fn apply_respects_existing_env_vars() {
        let prev = env::var("RAYON_NUM_THREADS").ok();
        env::set_var("RAYON_NUM_THREADS", "99");

        let config = ThreadConfig::compute(Some(4), None);
        config.apply();

        let val = env::var("RAYON_NUM_THREADS").unwrap();
        assert_eq!(val, "99");

        if let Some(v) = prev {
            env::set_var("RAYON_NUM_THREADS", v);
        } else {
            env::remove_var("RAYON_NUM_THREADS");
        }
    }

    #[test]
    fn thread_config_reasonable_defaults_for_common_core_counts() {
        for cores in [4, 8, 10, 12, 16, 32] {
            let budget = (cores * 75) / 100;
            let config = ThreadConfig::compute(Some(budget.max(2)), Some(CpuPreset::Auto));

            assert!(config.rayon_threads >= 2, "rayon >= 2 for {} cores", cores);
            assert!(config.onnx_threads >= 2, "onnx >= 2 for {} cores", cores);
            assert!(config.walker_threads <= 8, "walker <= 8 for {} cores", cores);
            assert!(
                config.rayon_threads + config.onnx_threads <= cores + 4,
                "total threads reasonable for {} cores",
                cores
            );
        }
    }
}
