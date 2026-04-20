window.DEMO_DATA = {
  "generated_at": "2026-04-05T18:58:45",
  "sources": {
    "experiment_record": "EXPERIMENT_RECORD.md",
    "window_report": "examples/kolmogorov_flow/comparison/re1e6_n512_ds4_latest_per_window_report.md",
    "pirate_config": "examples/kolmogorov_flow/configs/pirate.py",
    "soap_config": "examples/kolmogorov_flow/configs/paper_repro_soap.py",
    "legacy_full_window_summary": "eval_paper_repro_soap_0405/summary.txt",
    "localtime_full_window_summary": "eval_paper_repro_soap_0405_localtime/summary.txt",
    "window12_diagnostics_json": "eval_paper_repro_soap_0405_localtime/window12_vorticity_diagnostics.json"
  },
  "summary": {
    "key_message": "修正 eval time-axis 後，3137 的 u/v full-window 誤差維持低檔；真正仍需追蹤的是逐窗累積的 vorticity degradation。",
    "current_time_window": "12/50",
    "current_step": "60000/100000",
    "latest_checkpoint": "60000",
    "latest_w_err": 0.176233,
    "max_verified_window": 12,
    "u_err_mean": 0.002743,
    "v_err_mean": 0.002911,
    "w_err_mean": 0.060319,
    "legacy_w_err_mean": 0.437814,
    "window12_drop_factor": 3.634359058746092
  },
  "model": {
    "pirate": {
      "name": "pirate",
      "arch_name": "PirateNet",
      "num_layers": 4,
      "hidden_dim": 384,
      "out_dim": 3,
      "activation": "swish",
      "embed_dim": 384,
      "optimizer": "Adam",
      "batch_size_per_device": 4096,
      "num_time_windows": 10,
      "weighting_scheme": "grad_norm",
      "num_chunks": 16,
      "transfer_learning": true,
      "dataset_path": "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy",
      "gating_layers": 8
    },
    "soap": {
      "name": "paper_repro_soap",
      "arch_name": "PirateNet",
      "num_layers": 2,
      "hidden_dim": 768,
      "out_dim": 3,
      "activation": "swish",
      "embed_dim": 384,
      "optimizer": "Soap",
      "batch_size_per_device": 4096,
      "num_time_windows": 50,
      "weighting_scheme": "grad_norm",
      "num_chunks": 16,
      "transfer_learning": true,
      "dataset_path": "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy",
      "gating_layers": 4
    }
  },
  "reports": {
    "latest_final_step": [
      {
        "Window": 9,
        "Checkpoint": 60000,
        "t_eval": 0.85,
        "u_err": 0.003761,
        "v_err": 0.003802,
        "w_err": 0.10216
      }
    ],
    "latest_per_window": [
      {
        "Window": 1,
        "Latest Ckpt": 100000,
        "t_eval": 0.05,
        "u_err": 4.1e-05,
        "v_err": 4.2e-05,
        "w_err": 0.000266
      },
      {
        "Window": 2,
        "Latest Ckpt": 100000,
        "t_eval": 0.15,
        "u_err": 7.2e-05,
        "v_err": 7.1e-05,
        "w_err": 0.000857
      },
      {
        "Window": 3,
        "Latest Ckpt": 100000,
        "t_eval": 0.25,
        "u_err": 0.000151,
        "v_err": 0.000163,
        "w_err": 0.004119
      },
      {
        "Window": 4,
        "Latest Ckpt": 100000,
        "t_eval": 0.35,
        "u_err": 0.000328,
        "v_err": 0.000362,
        "w_err": 0.011118
      },
      {
        "Window": 5,
        "Latest Ckpt": 100000,
        "t_eval": 0.45,
        "u_err": 0.000615,
        "v_err": 0.000698,
        "w_err": 0.02181
      },
      {
        "Window": 6,
        "Latest Ckpt": 100000,
        "t_eval": 0.55,
        "u_err": 0.00106,
        "v_err": 0.001139,
        "w_err": 0.034955
      },
      {
        "Window": 7,
        "Latest Ckpt": 100000,
        "t_eval": 0.65,
        "u_err": 0.001642,
        "v_err": 0.001761,
        "w_err": 0.051751
      },
      {
        "Window": 8,
        "Latest Ckpt": 100000,
        "t_eval": 0.75,
        "u_err": 0.002495,
        "v_err": 0.00257,
        "w_err": 0.073567
      },
      {
        "Window": 9,
        "Latest Ckpt": 70000,
        "t_eval": 0.85,
        "u_err": 0.003701,
        "v_err": 0.003745,
        "w_err": 0.099886
      }
    ],
    "summary_statistics": [
      {
        "Metric": "u_err",
        "Mean": 0.002743,
        "Min": 0.000962,
        "Max": 0.007903
      },
      {
        "Metric": "v_err",
        "Mean": 0.002911,
        "Min": 0.001074,
        "Max": 0.008504
      },
      {
        "Metric": "w_err",
        "Mean": 0.060319,
        "Min": 0.000734,
        "Max": 0.176233
      }
    ],
    "window_report_results": [
      {
        "Window": 1,
        "Latest Ckpt": 100000,
        "t_eval": 0.05,
        "u_err": 4.1e-05,
        "v_err": 4.2e-05,
        "w_err": 0.000266
      },
      {
        "Window": 2,
        "Latest Ckpt": 100000,
        "t_eval": 0.15,
        "u_err": 7.2e-05,
        "v_err": 7.1e-05,
        "w_err": 0.000857
      },
      {
        "Window": 3,
        "Latest Ckpt": 100000,
        "t_eval": 0.25,
        "u_err": 0.000151,
        "v_err": 0.000163,
        "w_err": 0.004119
      },
      {
        "Window": 4,
        "Latest Ckpt": 100000,
        "t_eval": 0.35,
        "u_err": 0.000328,
        "v_err": 0.000362,
        "w_err": 0.011118
      },
      {
        "Window": 5,
        "Latest Ckpt": 100000,
        "t_eval": 0.45,
        "u_err": 0.000615,
        "v_err": 0.000698,
        "w_err": 0.02181
      },
      {
        "Window": 6,
        "Latest Ckpt": 100000,
        "t_eval": 0.55,
        "u_err": 0.00106,
        "v_err": 0.001139,
        "w_err": 0.034955
      },
      {
        "Window": 7,
        "Latest Ckpt": 100000,
        "t_eval": 0.65,
        "u_err": 0.001642,
        "v_err": 0.001761,
        "w_err": 0.051751
      },
      {
        "Window": 8,
        "Latest Ckpt": 100000,
        "t_eval": 0.75,
        "u_err": 0.002495,
        "v_err": 0.00257,
        "w_err": 0.073567
      },
      {
        "Window": 9,
        "Latest Ckpt": 70000,
        "t_eval": 0.85,
        "u_err": 0.003701,
        "v_err": 0.003745,
        "w_err": 0.099886
      }
    ],
    "verified_windows": [
      {
        "Window": 2,
        "Checkpoint": 40000,
        "t_eval": 0.15,
        "u_err": 0.000174,
        "v_err": 0.000177,
        "w_err": 0.001759
      },
      {
        "Window": 3,
        "Checkpoint": 40000,
        "t_eval": 0.25,
        "u_err": 0.000295,
        "v_err": 0.000322,
        "w_err": 0.00714
      }
    ],
    "current_losses": [
      {
        "name": "rc_loss",
        "range": "6.08e-05 ~ 6.51e-05"
      },
      {
        "name": "ru_loss",
        "range": "6.99e-05 ~ 8.27e-05"
      },
      {
        "name": "rv_loss",
        "range": "7.48e-05 ~ 1.04e-04"
      },
      {
        "name": "u_ic_loss",
        "range": "1.24e-07 ~ 1.44e-07"
      },
      {
        "name": "v_ic_loss",
        "range": "1.42e-07 ~ 2.02e-07"
      }
    ],
    "full_window_legacy": [
      {
        "Window": 1,
        "Checkpoint": 100000,
        "t_end": 0.05,
        "u_err": 0.001132,
        "v_err": 0.001147,
        "w_err": 0.000734
      },
      {
        "Window": 2,
        "Checkpoint": 100000,
        "t_end": 0.15,
        "u_err": 0.1126,
        "v_err": 0.118872,
        "w_err": 0.165103
      },
      {
        "Window": 3,
        "Checkpoint": 100000,
        "t_end": 0.25,
        "u_err": 0.145172,
        "v_err": 0.13748,
        "w_err": 0.254671
      },
      {
        "Window": 4,
        "Checkpoint": 100000,
        "t_end": 0.35,
        "u_err": 0.159799,
        "v_err": 0.161813,
        "w_err": 0.349038
      },
      {
        "Window": 5,
        "Checkpoint": 100000,
        "t_end": 0.45,
        "u_err": 0.154807,
        "v_err": 0.188169,
        "w_err": 0.429782
      },
      {
        "Window": 6,
        "Checkpoint": 100000,
        "t_end": 0.55,
        "u_err": 0.147691,
        "v_err": 0.188783,
        "w_err": 0.482639
      },
      {
        "Window": 7,
        "Checkpoint": 100000,
        "t_end": 0.65,
        "u_err": 0.157871,
        "v_err": 0.177243,
        "w_err": 0.519076
      },
      {
        "Window": 8,
        "Checkpoint": 100000,
        "t_end": 0.75,
        "u_err": 0.167141,
        "v_err": 0.173145,
        "w_err": 0.557658
      },
      {
        "Window": 9,
        "Checkpoint": 100000,
        "t_end": 0.85,
        "u_err": 0.189373,
        "v_err": 0.168829,
        "w_err": 0.590839
      },
      {
        "Window": 10,
        "Checkpoint": 100000,
        "t_end": 0.95,
        "u_err": 0.190959,
        "v_err": 0.194211,
        "w_err": 0.627286
      },
      {
        "Window": 11,
        "Checkpoint": 100000,
        "t_end": 1.05,
        "u_err": 0.186167,
        "v_err": 0.193459,
        "w_err": 0.636445
      },
      {
        "Window": 12,
        "Checkpoint": 60000,
        "t_end": 1.15,
        "u_err": 0.18039,
        "v_err": 0.178034,
        "w_err": 0.640494
      }
    ],
    "full_window_localtime": [
      {
        "Window": 1,
        "Checkpoint": 100000,
        "t_end": 0.05,
        "u_err": 0.001132,
        "v_err": 0.001147,
        "w_err": 0.000734
      },
      {
        "Window": 2,
        "Checkpoint": 100000,
        "t_end": 0.15,
        "u_err": 0.000962,
        "v_err": 0.001129,
        "w_err": 0.001016
      },
      {
        "Window": 3,
        "Checkpoint": 100000,
        "t_end": 0.25,
        "u_err": 0.001006,
        "v_err": 0.001074,
        "w_err": 0.00339
      },
      {
        "Window": 4,
        "Checkpoint": 100000,
        "t_end": 0.35,
        "u_err": 0.001003,
        "v_err": 0.001104,
        "w_err": 0.009881
      },
      {
        "Window": 5,
        "Checkpoint": 100000,
        "t_end": 0.45,
        "u_err": 0.001058,
        "v_err": 0.001145,
        "w_err": 0.020085
      },
      {
        "Window": 6,
        "Checkpoint": 100000,
        "t_end": 0.55,
        "u_err": 0.001335,
        "v_err": 0.00147,
        "w_err": 0.033546
      },
      {
        "Window": 7,
        "Checkpoint": 100000,
        "t_end": 0.65,
        "u_err": 0.001789,
        "v_err": 0.001823,
        "w_err": 0.049322
      },
      {
        "Window": 8,
        "Checkpoint": 100000,
        "t_end": 0.75,
        "u_err": 0.002464,
        "v_err": 0.002574,
        "w_err": 0.070874
      },
      {
        "Window": 9,
        "Checkpoint": 100000,
        "t_end": 0.85,
        "u_err": 0.003507,
        "v_err": 0.003539,
        "w_err": 0.093511
      },
      {
        "Window": 10,
        "Checkpoint": 100000,
        "t_end": 0.95,
        "u_err": 0.004674,
        "v_err": 0.004896,
        "w_err": 0.11963
      },
      {
        "Window": 11,
        "Checkpoint": 100000,
        "t_end": 1.05,
        "u_err": 0.006081,
        "v_err": 0.006522,
        "w_err": 0.145606
      },
      {
        "Window": 12,
        "Checkpoint": 60000,
        "t_end": 1.15,
        "u_err": 0.007903,
        "v_err": 0.008504,
        "w_err": 0.176233
      }
    ],
    "full_window_rerun_delta": [
      {
        "Window": 1,
        "legacy_w_err": 0.000734,
        "localtime_w_err": 0.000734,
        "drop_factor": 1.0,
        "legacy_uv_mean": 0.0011394999999999999,
        "localtime_uv_mean": 0.0011394999999999999
      },
      {
        "Window": 2,
        "legacy_w_err": 0.165103,
        "localtime_w_err": 0.001016,
        "drop_factor": 162.5029527559055,
        "legacy_uv_mean": 0.115736,
        "localtime_uv_mean": 0.0010455
      },
      {
        "Window": 3,
        "legacy_w_err": 0.254671,
        "localtime_w_err": 0.00339,
        "drop_factor": 75.12418879056047,
        "legacy_uv_mean": 0.141326,
        "localtime_uv_mean": 0.0010400000000000001
      },
      {
        "Window": 4,
        "legacy_w_err": 0.349038,
        "localtime_w_err": 0.009881,
        "drop_factor": 35.32415747393989,
        "legacy_uv_mean": 0.160806,
        "localtime_uv_mean": 0.0010535
      },
      {
        "Window": 5,
        "legacy_w_err": 0.429782,
        "localtime_w_err": 0.020085,
        "drop_factor": 21.39815782922579,
        "legacy_uv_mean": 0.171488,
        "localtime_uv_mean": 0.0011015
      },
      {
        "Window": 6,
        "legacy_w_err": 0.482639,
        "localtime_w_err": 0.033546,
        "drop_factor": 14.387378525010433,
        "legacy_uv_mean": 0.168237,
        "localtime_uv_mean": 0.0014025
      },
      {
        "Window": 7,
        "legacy_w_err": 0.519076,
        "localtime_w_err": 0.049322,
        "drop_factor": 10.524228538988687,
        "legacy_uv_mean": 0.167557,
        "localtime_uv_mean": 0.0018059999999999999
      },
      {
        "Window": 8,
        "legacy_w_err": 0.557658,
        "localtime_w_err": 0.070874,
        "drop_factor": 7.868301492790021,
        "legacy_uv_mean": 0.170143,
        "localtime_uv_mean": 0.002519
      },
      {
        "Window": 9,
        "legacy_w_err": 0.590839,
        "localtime_w_err": 0.093511,
        "drop_factor": 6.3183903497984195,
        "legacy_uv_mean": 0.179101,
        "localtime_uv_mean": 0.003523
      },
      {
        "Window": 10,
        "legacy_w_err": 0.627286,
        "localtime_w_err": 0.11963,
        "drop_factor": 5.2435509487586724,
        "legacy_uv_mean": 0.192585,
        "localtime_uv_mean": 0.004785
      },
      {
        "Window": 11,
        "legacy_w_err": 0.636445,
        "localtime_w_err": 0.145606,
        "drop_factor": 4.3710080628545525,
        "legacy_uv_mean": 0.189813,
        "localtime_uv_mean": 0.0063015
      },
      {
        "Window": 12,
        "legacy_w_err": 0.640494,
        "localtime_w_err": 0.176233,
        "drop_factor": 3.634359058746092,
        "legacy_uv_mean": 0.17921199999999998,
        "localtime_uv_mean": 0.008203499999999999
      }
    ],
    "window12_diagnostics": {
      "window": 12,
      "checkpoint": 60000,
      "t_end": 1.1500000000000001,
      "mean_w_err": 0.17622988777735132,
      "final_w_err": 0.17904606753463684,
      "peak_w_err": 0.17904606753463684,
      "peak_w_err_time": 1.1500000000000001,
      "final_corr": 0.9838434861252203,
      "final_bias": 0.0008485599952054584,
      "final_std_ratio": 0.981498983454875,
      "final_enstrophy_ratio": 0.9633402422610367,
      "low_k_ratio": 0.9995272195164485,
      "high_k_ratio": 0.6646327956630084
    }
  }
};
