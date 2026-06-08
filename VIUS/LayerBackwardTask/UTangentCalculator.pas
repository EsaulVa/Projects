unit UTangentCalculator;

interface

{$REGION ' uses '}
uses // ¬ ¿À‘¿¬»“ÕŒÃ œŒ–ﬂƒ ≈!
{$I '..\..\..\Common Files\GeneralUses.pas'},
  Classes;
{$ENDREGION ' uses '}

type

  TTangentCalculator = class(TTunable, ITangentCalculator)
    public
      function Calculate(const Z: Extended; const Surface: IRegularSurface;
        const ToGR: RVector3; var WP: RWindPoint): Boolean; virtual; abstract;
  end;

  TTangentCalculatorI = class(TTangentCalculator)
    public
      function Calculate(const Z: Extended; const Surface: IRegularSurface;
        const ToGR: RVector3; var WP: RWindPoint): Boolean; override;
  end;

implementation

{ TTangentCalculatorI }

function TTangentCalculatorI.Calculate(const Z: Extended; const Surface: IRegularSurface;
  const ToGR: RVector3; var WP: RWindPoint): Boolean;
var
  wu, wv, tau: RVector3;
  denomerator: Extended;
begin
  Result := False;
  try
    with WP do
    begin
      tau := (ToGR - Surface.R(U, V)).Normalize;
      wu := Surface.Ru(U, V);
      wv := Surface.Rv(U, V);
    end;
    denomerator := wu.Sqr * wv.Sqr - Sqr(wu * wv);
    wp.dU := (wv.Sqr * (wu * tau) - (wu * wv) * (wv * tau)) / denomerator;
    wp.dV := (wu.Sqr * (wv * tau) - (wu * wv) * (wu * tau)) / denomerator;
    Result := True;
  except
  end;
end;

initialization
  RegisterClass(TTangentCalculatorI);

end.
