class Logger:
    @staticmethod
    def training_log(epoch_idx, max_epoch, avg_loss, accuracy):
        print(f"epoch {epoch_idx+1}/{max_epoch}, average loss: {avg_loss}, accuracy: {accuracy}")