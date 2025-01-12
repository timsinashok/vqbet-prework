# Pre-work for CAIR Lab

### Preliminaries:
- ACT repo: https://github.com/tonyzhaozh/act
- VQ-BET repo: https://github.com/jayLEE0301/vq_bet_official

These two are policy learning ML techniques in robotics. If you can train vq-bet policy on aloha environment (dataset given on point 3 below) and save the result, that will be an important milestone in understanding robot policy learning. Here is the a simple guide to help you:
1. Read ACT paper, read VQ-BET paper
2. Go through ACT repo, ALOHA repo and VQ-BET repo
3. Download the simulated human demonstration dataset from google drive provided in act repo, either cube transfer or insert.
4. Train on VQ-BET model
5. Evaluate and save video results.

`Helpful hints`: you can try training ACT policy to see how it is trained, then move to VQ-BET. Take your time and after understanding the codebase, it should be straightforward to train, evaluate in the simulation and save the video of rollout. If you have a GPU laptop, you can do it locally provided you have linux, if not after you are familiar with everything and do simple code modifications, you can request for a server with professor. Please make sure to brush up on basic linux command line skills. 