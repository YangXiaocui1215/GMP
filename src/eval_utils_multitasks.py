import torch
import torch.nn as nn


def eval(args, model, loader, metric, device):
    num_correct =0 
    model.eval()
    for i, batch in enumerate(loader):
        # Forward pass
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}
        # import ipdb; ipdb.set_trace()
        predict, predict_aspects_num = model.predict(
            input_ids=batch['input_ids'].to(device),
            image_features=list(
                map(lambda x: x.to(device), batch['image_features'])),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos, 
            aspects_num=batch['aspects_num'])
        target_aspects_num = torch.tensor(batch['aspects_num']).to(predict_aspects_num.device)
        num_correct += torch.eq(predict_aspects_num, target_aspects_num).sum().float().item()
        
        # print('predict is {}'.format(predict))

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break
    aspects_num_eval_acc = num_correct/len(loader.dataset)
    res = metric.get_metric()
    model.train()
    return res, aspects_num_eval_acc
